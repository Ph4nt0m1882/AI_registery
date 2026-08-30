import json
import os
import sys
from datetime import datetime
from pathlib import Path
from google import genai

# Chargement automatique du .env local ou parent
if not os.getenv("GEMINI_API_KEY"):
    for env_file in [Path(".env"), Path("../.env"), Path(__file__).resolve().parent.parent / ".env"]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"\''))
            break

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    API_KEY = input("Veuillez entrer votre clé API Gemini : ")

client = genai.Client(api_key=API_KEY)

# Taxonomie standardisée par tâche
DEFAULT_TASKS = [
    {
        "id": "text-generation",
        "name": "Text Generation (Chat, Raisonnement, Code)",
        "default_inputs": ["text", "image", "audio", "video", "pdf"],
        "default_outputs": ["text"]
    },
    {
        "id": "text-to-image",
        "name": "Text-to-Image (Génération d'images)",
        "default_inputs": ["text", "image"],
        "default_outputs": ["image"]
    },
    {
        "id": "image-to-image",
        "name": "Image-to-Image (Retouche, Inpainting, Variation)",
        "default_inputs": ["image", "text"],
        "default_outputs": ["image"]
    },
    {
        "id": "text-to-video",
        "name": "Text-to-Video (Génération de vidéos)",
        "default_inputs": ["text", "image", "video"],
        "default_outputs": ["video"]
    },
    {
        "id": "text-to-music",
        "name": "Text-to-Music (Génération & Composition musicale)",
        "default_inputs": ["text", "audio"],
        "default_outputs": ["audio"]
    },
    {
        "id": "text-to-speech",
        "name": "Text-to-Speech (Synthèse vocale / Voix)",
        "default_inputs": ["text"],
        "default_outputs": ["audio"]
    },
    {
        "id": "speech-to-text",
        "name": "Speech-to-Text (Transcription & Reconnaissance vocale)",
        "default_inputs": ["audio"],
        "default_outputs": ["text"]
    },
    {
        "id": "realtime-multimodal",
        "name": "Realtime Multimodal (Live audio/vidéo bidirectionnel)",
        "default_inputs": ["audio", "text", "video"],
        "default_outputs": ["audio", "text"]
    },
    {
        "id": "deep-research",
        "name": "Deep Research & Agents (Recherche web autonome, Computer-use)",
        "default_inputs": ["text", "image"],
        "default_outputs": ["text"]
    },
    {
        "id": "embeddings",
        "name": "Embeddings (Vectorisation & Recherche sémantique / RAG)",
        "default_inputs": ["text"],
        "default_outputs": ["embeddings"]
    },
]

def detect_modalities_and_task_hint(model):
    """Détecte les modalités (inputs/outputs) et suggère la tâche appropriée."""
    name_lower = model.name.lower()
    desc_lower = (model.description or "").lower()
    actions = model.supported_actions or []
    
    # Par défaut pour la famille Gemini
    suggested_task_id = "text-generation"
    input_modalities = ["text"]
    output_modalities = ["text"]
    
    # 1. Vidéo (Veo)
    if "veo" in name_lower or "predictLongRunning" in actions:
        suggested_task_id = "text-to-video"
        input_modalities = ["text", "image", "video"]
        output_modalities = ["video"]
        
    # 2. Musique (Lyria)
    elif "lyria" in name_lower or "music" in desc_lower:
        suggested_task_id = "text-to-music"
        input_modalities = ["text", "audio"]
        output_modalities = ["audio"]
        
    # 3. Synthèse Vocale (TTS)
    elif "tts" in name_lower:
        suggested_task_id = "text-to-speech"
        input_modalities = ["text"]
        output_modalities = ["audio"]
        
    # 4. Image (Imagen, Gemini Image, Nano Banana)
    elif "image" in name_lower or "imagen" in name_lower or "banana" in name_lower:
        suggested_task_id = "text-to-image"
        input_modalities = ["text", "image"]
        output_modalities = ["image"]
        
    # 5. Transcription audio
    elif "transcribe" in name_lower and "live" not in name_lower:
        suggested_task_id = "speech-to-text"
        input_modalities = ["audio"]
        output_modalities = ["text"]
        
    # 6. Live / Realtime
    elif "bidiGenerateContent" in actions or "native-audio" in name_lower or "live" in name_lower:
        suggested_task_id = "realtime-multimodal"
        input_modalities = ["audio", "text", "video"]
        output_modalities = ["audio", "text"]
        
    # 7. Deep Research & Agents
    elif "deep-research" in name_lower or "antigravity" in name_lower or "computer-use" in name_lower or "robotics" in name_lower:
        suggested_task_id = "deep-research"
        input_modalities = ["text", "image"]
        output_modalities = ["text"]
        
    # 8. Embeddings
    elif "embedding" in name_lower or "embedContent" in actions:
        suggested_task_id = "embeddings"
        input_modalities = ["text", "image", "audio", "video"] if "embedding-2" in name_lower else ["text"]
        output_modalities = ["embeddings"]
        
    # 9. LLM Multimodal standard (Gemini Flash, Pro, Omni)
    elif "gemini" in name_lower:
        suggested_task_id = "text-generation"
        input_modalities = ["text", "image", "audio", "video", "pdf"]
        output_modalities = ["text"]
        
    # 10. Gemma (Texte pur)
    elif "gemma" in name_lower:
        suggested_task_id = "text-generation"
        input_modalities = ["text"]
        output_modalities = ["text"]

    is_multimodal = len(set(input_modalities + output_modalities) - {"text", "embeddings"}) > 0 or len(input_modalities) > 1
    
    return suggested_task_id, input_modalities, output_modalities, is_multimodal

def load_existing_registry(json_path: Path):
    """Charge les classifications existantes."""
    model_to_info = {}
    tasks_map = {t["id"]: t["name"] for t in DEFAULT_TASKS}
    
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "tasks" in data and isinstance(data["tasks"], dict):
                for task_id, models in data["tasks"].items():
                    for m in models:
                        if isinstance(m, dict):
                            full_id = m.get("full_id") or f"models/{m.get('id')}"
                            short_id = m.get("id") or full_id.replace("models/", "")
                            model_to_info[full_id] = m
                            model_to_info[short_id] = m
        except Exception as e:
            print(f"⚠️ Note : Initialisation d'un nouveau registre ({e}).")
            
    return model_to_info, tasks_map

def prompt_task_for_model(model, tasks_map):
    """Affiche la fiche du modèle et propose le choix de la tâche."""
    sugg_task_id, in_mods, out_mods, is_multi = detect_modalities_and_task_hint(model)
    
    print("\n" + "=" * 68)
    print(" 🔍 NOUVEAU MODÈLE À CLASSER")
    print("=" * 68)
    print(f" ID             : {model.name}")
    print(f" Nom            : {model.display_name or 'N/A'}")
    print(f" Description    : {model.description or 'N/A'}")
    tokens_in = f"{model.input_token_limit:,}" if model.input_token_limit else "N/A"
    tokens_out = f"{model.output_token_limit:,}" if model.output_token_limit else "N/A"
    print(f" Tokens         : {tokens_in} in  /  {tokens_out} out")
    print(f" Actions        : {model.supported_actions or []}")
    print(f" 💡 Modalités   : Input {in_mods} ➔ Output {out_mods} (Multimodal: {is_multi})")
    print(f" 💡 Suggestion  : {tasks_map.get(sugg_task_id, sugg_task_id)}")
    print("-" * 68)
    print("Sélectionnez la tâche principale pour ce modèle :")
    
    tasks_keys = list(tasks_map.keys())
    sugg_index = 1
    for idx, t_id in enumerate(tasks_keys, 1):
        is_sugg = "  👈 [RECOMMANDÉ]" if t_id == sugg_task_id else ""
        if t_id == sugg_task_id:
            sugg_index = idx
        print(f"  [{idx}] {tasks_map[t_id]} (`{t_id}`){is_sugg}")
        
    print(f"  [+] Créer une nouvelle tâche personnalisée")
    print(f"  [Entrée directe] Appuyez sur Entrée pour valider le choix recommandé [{sugg_index}]")
    print("-" * 68)
    
    while True:
        choice = input(f"👉 Votre choix [{sugg_index}] : ").strip()
        
        # Validation par défaut de la recommandation si l'utilisateur appuie juste sur Entrée
        if not choice:
            chosen_task_id = sugg_task_id
            return chosen_task_id, in_mods, out_mods, is_multi
            
        if choice == '+':
            new_task_id = input("✨ Entrez l'identifiant technique de la tâche (ex: robotics-control) : ").strip().lower().replace(" ", "-")
            new_task_name = input("✨ Entrez le libellé descriptif (ex: Robotics & Hardware Control) : ").strip()
            if new_task_id and new_task_name:
                tasks_map[new_task_id] = new_task_name
                return new_task_id, in_mods, out_mods, is_multi
            print("❌ Tâche invalide. Réessayez.")
            continue
            
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(tasks_keys):
                chosen_task_id = tasks_keys[idx - 1]
                return chosen_task_id, in_mods, out_mods, is_multi
            print(f"❌ Numéro invalide (doit être entre 1 et {len(tasks_keys)}).")
            continue
            
        # Si tapé directement
        if choice in tasks_map:
            return choice, in_mods, out_mods, is_multi

import re

def score_model_for_ranking(model: dict):
    """Calcule un score de pertinence pour trier les modèles du plus récent/recommandé au plus ancien."""
    model_id = model["id"].lower()
    
    # 1. Famille (Gemini > Gemma > AQA / Autre)
    family_score = 3
    if model_id.startswith("gemma"):
        family_score = 2
    elif model_id == "aqa":
        family_score = 1
        
    # 2. Date pour les modèles datés (ex: deep-research-max-preview-04-2026 -> 2026.33)
    date_match = re.search(r'(\d{2})-(\d{4})', model_id)
    date_score = 0.0
    model_id_no_date = model_id
    if date_match:
        month, year = int(date_match.group(1)), int(date_match.group(2))
        date_score = year + (month / 12.0)
        model_id_no_date = model_id.replace(date_match.group(0), '')
        
    # 3. Numéro de version réel (ex: 3.7, 3.6, 3.5, 3.1, 2.5)
    version_match = re.search(r'(\d+(?:\.\d+)?)', model_id_no_date)
    version = float(version_match.group(1)) if version_match else (date_score if date_score > 0 else 1.0)
        
    # 4. Alias '-latest' (souvent le pointeur vers la version stable par excellence)
    is_latest = 1 if "-latest" in model_id else 0
    
    # 5. Gamme de puissance (Max > Pro > Flash > Lite > Fast > Autre)
    tier_weight = 3
    if "max" in model_id:
        tier_weight = 5
    elif "pro" in model_id and "image" not in model_id:
        tier_weight = 4
    elif "flash" in model_id and "lite" not in model_id:
        tier_weight = 3
    elif "fast" in model_id:
        tier_weight = 2
    elif "lite" in model_id:
        tier_weight = 1
        
    is_standard = 1 if ("lite" not in model_id and "fast" not in model_id and "customtools" not in model_id) else 0

    return (family_score, version, date_score, is_latest, tier_weight, is_standard, model["id"])

def save_registry(json_path: Path, categorized_tasks: dict, total_count: int):
    """Sauvegarde le catalogue unifié au format JSON structuré par tâche et ordonné par pertinence."""
    sorted_tasks = {}
    for task_id, models in categorized_tasks.items():
        # Tri décroissant du meilleur/plus récent au plus ancien
        sorted_list = sorted(models, key=score_model_for_ranking, reverse=True)
        # Attribution du rank et du flag is_recommended
        for rank, m in enumerate(sorted_list, 1):
            m["rank"] = rank
            m["is_recommended"] = (rank == 1)
        sorted_tasks[task_id] = sorted_list

    output_data = {
        "schema_version": "1.0.0",
        "provider": "google_ai_studio",
        "name": "Google AI Studio (Gemini)",
        "updated_at": datetime.now().isoformat(),
        "total_models": total_count,
        "tasks_summary": {k: len(v) for k, v in sorted_tasks.items()},
        "tasks": sorted_tasks
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    json_path = Path(__file__).resolve().parent / "google_ai_studio.json"
    reclassify_all = "--reset" in sys.argv or "--reclassify" in sys.argv
    
    if reclassify_all:
        print("🔄 Mode reclassement complet (--reset).")
        model_to_info = {}
        tasks_map = {t["id"]: t["name"] for t in DEFAULT_TASKS}
    else:
        model_to_info, tasks_map = load_existing_registry(json_path)
    
    print("📡 Récupération des modèles depuis Google AI Studio...")
    live_models = list(client.models.list())
    print(f"✅ {len(live_models)} modèles récupérés au total.")
    
    categorized_tasks = {}
    new_classified = 0
    
    for model in live_models:
        model_key = model.name
        short_key = model.name.replace("models/", "")
        
        if (model_key in model_to_info or short_key in model_to_info) and not reclassify_all:
            existing = model_to_info.get(model_key) or model_to_info.get(short_key)
            task_id = existing.get("task", "text-generation")
            in_mods = existing.get("input_modalities", ["text"])
            out_mods = existing.get("output_modalities", ["text"])
            is_multi = existing.get("is_multimodal", True)
        else:
            new_classified += 1
            task_id, in_mods, out_mods, is_multi = prompt_task_for_model(model, tasks_map)
            
        # Détermination du statut de mise en avant (Featured / Standard / Specialized / Legacy)
        is_featured = short_key in {
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro-preview",
            "deep-research-max-preview-04-2026",
            "gemma-4-31b-it",
            "veo-3.1-generate-preview",
            "gemini-3.1-flash-image",
            "lyria-3-pro-preview",
            "gemini-3.1-flash-tts-preview",
            "gemini-robotics-er-2-streaming-preview"
        }
        
        tier = "standard"
        if is_featured:
            tier = "featured"
        elif "customtools" in short_key or "computer-use" in short_key or short_key == "aqa" or "robotics" in short_key:
            tier = "specialized"
        elif "2.5" in short_key or "preview-12-2025" in short_key or "3-flash-preview" in short_key:
            tier = "legacy"

        entry = {
            "id": short_key,
            "full_id": model.name,
            "display_name": model.display_name,
            "description": model.description,
            "task": task_id,
            "tier": tier,
            "is_featured": is_featured,
            "is_multimodal": is_multi,
            "input_modalities": in_mods,
            "output_modalities": out_mods,
            "context_window": model.input_token_limit,
            "max_output_tokens": model.output_token_limit,
            "supports_thinking": getattr(model, "thinking", None) is not None,
            "supported_actions": model.supported_actions,
            "temperature": getattr(model, "temperature", None),
        }
        
        categorized_tasks.setdefault(task_id, []).append(entry)
        save_registry(json_path, categorized_tasks, len(live_models))
        
    print("\n" + "=" * 68)
    print(" 🎉 REGISTRE IA MIS À JOUR ET ENREGISTRÉ AVEC SUCCÈS")
    print("=" * 68)
    print(f"📊 Modèles totaux : {len(live_models)}")
    print(f"🆕 Nouveaux triés : {new_classified}")
    print(f"📁 Fichier JSON   : {json_path}")
    print("\nRésumé par tâche :")
    for t_id, models in categorized_tasks.items():
        print(f"  • {tasks_map.get(t_id, t_id)} (`{t_id}`) : {len(models)} modèle(s)")
    print("=" * 68)

if __name__ == "__main__":
    main()