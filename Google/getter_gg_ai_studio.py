import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Chargement automatique du .env local ou parent
if not os.getenv("GEMINI_API_KEY"):
    for env_file in [
        Path(".env"),
        Path("../.env"),
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"\''))
            break

API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_TASKS = [
    {
        "id": "text-generation",
        "name": "Text Generation (Chat, Raisonnement, Code)",
        "default_inputs": ["text", "image", "audio", "video"],
        "default_outputs": ["text"],
        "supports_documents": True,
    },
    {
        "id": "text-to-image",
        "name": "Text-to-Image (Génération d'images)",
        "default_inputs": ["text", "image"],
        "default_outputs": ["image"],
        "supports_documents": False,
    },
    {
        "id": "image-to-image",
        "name": "Image-to-Image (Retouche, Inpainting, Variation)",
        "default_inputs": ["image", "text"],
        "default_outputs": ["image"],
        "supports_documents": False,
    },
    {
        "id": "text-to-video",
        "name": "Text-to-Video (Génération de vidéos)",
        "default_inputs": ["text", "image", "video"],
        "default_outputs": ["video"],
        "supports_documents": False,
    },
    {
        "id": "text-to-music",
        "name": "Text-to-Music (Génération & Composition musicale)",
        "default_inputs": ["text", "audio"],
        "default_outputs": ["audio"],
        "supports_documents": False,
    },
    {
        "id": "text-to-speech",
        "name": "Text-to-Speech (Synthèse vocale / Voix)",
        "default_inputs": ["text"],
        "default_outputs": ["audio"],
        "supports_documents": False,
    },
    {
        "id": "speech-to-text",
        "name": "Speech-to-Text (Transcription & Reconnaissance vocale)",
        "default_inputs": ["audio"],
        "default_outputs": ["text"],
        "supports_documents": False,
    },
    {
        "id": "realtime-multimodal",
        "name": "Realtime Multimodal (Live audio/vidéo bidirectionnel)",
        "default_inputs": ["audio", "text", "video"],
        "default_outputs": ["audio", "text"],
        "supports_documents": False,
    },
    {
        "id": "deep-research",
        "name": "Deep Research & Agents (Recherche web autonome, Computer-use)",
        "default_inputs": ["text", "image"],
        "default_outputs": ["text"],
        "supports_documents": True,
    },
    {
        "id": "robotics",
        "name": "Robotics & Hardware Control",
        "default_inputs": ["text", "image", "video"],
        "default_outputs": ["text"],
        "supports_documents": False,
    },
    {
        "id": "embeddings",
        "name": "Embeddings (Vectorisation & Recherche sémantique / RAG)",
        "default_inputs": ["text"],
        "default_outputs": ["embeddings"],
        "supports_documents": False,
    },
]

TASKS_MAP = {t["id"]: t["name"] for t in DEFAULT_TASKS}

FEATURED_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "deep-research-max-preview-04-2026",
    "gemma-4-31b-it",
    "veo-3.1-generate-preview",
    "gemini-3.1-flash-image",
    "lyria-3-pro-preview",
    "gemini-3.1-flash-tts-preview",
    "gemini-robotics-er-2-streaming-preview",
    "gemini-3.8-live",
}


def infer_model_features(short_id: str, supported_actions: list, supports_thinking: bool) -> dict:
    """Déduit les capacités fonctionnelles (features) du modèle."""
    name_lower = short_id.lower()
    actions = supported_actions or []

    is_content_gen = "generateContent" in actions or "bidiGenerateContent" in actions
    is_specialized = any(
        x in name_lower
        for x in ["tts", "music", "lyria", "embedding", "veo", "banana", "imagen", "transcribe", "stt"]
    )

    thinking = supports_thinking or ("thinking" in name_lower)
    function_calling = is_content_gen and not is_specialized
    structured_outputs = is_content_gen and not is_specialized
    system_instruction = is_content_gen and not any(x in name_lower for x in ["embedding", "music", "lyria", "veo"])
    caching = "createCachedContent" in actions
    streaming = "generateContent" in actions or "bidiGenerateContent" in actions

    return {
        "thinking": thinking,
        "function_calling": function_calling,
        "structured_outputs": structured_outputs,
        "system_instruction": system_instruction,
        "caching": caching,
        "streaming": streaming,
    }


def detect_modalities_and_task_hint(short_id: str, desc: str = "", actions: list = None):
    """Détecte les modalités (inputs/outputs) et suggère la tâche principale."""
    name_lower = short_id.lower()
    desc_lower = (desc or "").lower()
    actions = actions or []

    suggested_task_id = "text-generation"
    input_modalities = ["text"]
    output_modalities = ["text"]
    supports_documents = False

    if "veo" in name_lower or "predictLongRunning" in actions:
        suggested_task_id = "text-to-video"
        input_modalities = ["text", "image", "video"]
        output_modalities = ["video"]
    elif "lyria" in name_lower or "music" in desc_lower:
        suggested_task_id = "text-to-music"
        input_modalities = ["text", "audio"]
        output_modalities = ["audio"]
    elif "tts" in name_lower:
        suggested_task_id = "text-to-speech"
        input_modalities = ["text"]
        output_modalities = ["audio"]
    elif "image" in name_lower or "imagen" in name_lower or "banana" in name_lower:
        suggested_task_id = "text-to-image"
        input_modalities = ["text", "image"]
        output_modalities = ["image"]
    elif "transcribe" in name_lower and "live" not in name_lower:
        suggested_task_id = "speech-to-text"
        input_modalities = ["audio"]
        output_modalities = ["text"]
    elif "bidiGenerateContent" in actions or "native-audio" in name_lower or "live" in name_lower:
        suggested_task_id = "realtime-multimodal"
        input_modalities = ["audio", "text", "video"]
        output_modalities = ["audio", "text"]
    elif "robotics" in name_lower:
        suggested_task_id = "robotics"
        input_modalities = ["text", "image", "video"]
        output_modalities = ["text"]
    elif (
        "deep-research" in name_lower
        or "antigravity" in name_lower
        or "computer-use" in name_lower
    ):
        suggested_task_id = "deep-research"
        input_modalities = ["text", "image"]
        output_modalities = ["text"]
        supports_documents = True
    elif "embedding" in name_lower or "embedContent" in actions:
        suggested_task_id = "embeddings"
        input_modalities = ["text", "image", "audio", "video"] if "embedding-2" in name_lower else ["text"]
        output_modalities = ["embeddings"]
    elif "gemini" in name_lower:
        suggested_task_id = "text-generation"
        input_modalities = ["text", "image", "audio", "video"]
        output_modalities = ["text"]
        supports_documents = True
    elif "gemma" in name_lower:
        suggested_task_id = "text-generation"
        input_modalities = ["text"]
        output_modalities = ["text"]

    return suggested_task_id, input_modalities, output_modalities, supports_documents


def infer_category(short_id: str) -> str:
    """Déduit une catégorie pour le modèle."""
    name_lower = short_id.lower()
    if "flash" in name_lower and "lite" in name_lower:
        return "flash_lite"
    if "flash" in name_lower and "image" in name_lower:
        return "image_gen"
    if "flash" in name_lower:
        return "flash"
    if "pro" in name_lower and "image" in name_lower:
        return "image_gen"
    if "pro" in name_lower:
        return "pro"
    if "live" in name_lower:
        return "live"
    if "thinking" in name_lower:
        return "thinking"
    if "gemma" in name_lower:
        return "open_source"
    if "veo" in name_lower:
        return "video_gen"
    if "lyria" in name_lower:
        return "music"
    if "tts" in name_lower:
        return "tts"
    if "transcribe" in name_lower:
        return "transcribe"
    if "deep-research" in name_lower or "research" in name_lower:
        return "research"
    if "antigravity" in name_lower:
        return "agent"
    if "robotics" in name_lower:
        return "robotics"
    if "embedding" in name_lower:
        return "embedding"
    return "general"


def infer_tier(short_id: str) -> str:
    """Déduit le tier (featured, standard, preview, specialized, legacy)."""
    name_lower = short_id.lower()
    if short_id in FEATURED_MODELS:
        return "featured"
    if any(x in name_lower for x in ["customtools", "computer-use", "robotics"]) or short_id == "aqa":
        return "specialized"
    if any(x in name_lower for x in ["2.5", "preview-12-2025", "3-flash-preview", "1.5"]):
        return "legacy"
    if "preview" in name_lower:
        return "preview"
    return "standard"


def score_model_for_ranking(model_card: dict):
    """Calcule un score de pertinence pour ordonner les modèles du plus recommandé/récent au plus ancien."""
    model_id = model_card["id"].lower()

    # 1. Poids du Tier (Featured en premier !)
    tier = model_card.get("tier", "standard")
    if model_card["id"] in FEATURED_MODELS:
        tier = "featured"
    tier_weights = {"featured": 5, "standard": 4, "preview": 3, "specialized": 2, "legacy": 1}
    tier_weight = tier_weights.get(tier, 3)

    # 2. Famille (Gemini > Gemma > AQA / Autre)
    family_score = 3
    if model_id.startswith("gemma"):
        family_score = 2
    elif model_id == "aqa":
        family_score = 1

    # 3. Date pour les modèles datés
    date_match = re.search(r"(\d{2})-(\d{4})", model_id)
    date_score = 0.0
    model_id_no_date = model_id
    if date_match:
        month, year = int(date_match.group(1)), int(date_match.group(2))
        date_score = year + (month / 12.0)
        model_id_no_date = model_id.replace(date_match.group(0), "")

    # 4. Numéro de version (3.8 > 3.7 > 3.5 > 3.1 > 3.0 > 2.5)
    version_match = re.search(r"(\d+(?:\.\d+)?)", model_id_no_date)
    version = float(version_match.group(1)) if version_match else 1.0

    # 5. Alias '-latest'
    is_latest = 1 if "-latest" in model_id else 0

    return (tier_weight, family_score, version, date_score, is_latest, model_card["id"])


def load_existing_registry(json_path: Path):
    """
    Charge le registre précédent (supporte le format V1 avec 'tasks' ou le format V2 avec 'models').
    Retourne (existing_models_dict, quick_picks).
    """
    existing_models = {}
    quick_picks = {}

    if not json_path.exists():
        return existing_models, quick_picks

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        quick_picks = data.get("quick_picks", {})

        # Cas 1 : Format V2 existant (table plate 'models')
        if "models" in data and isinstance(data["models"], dict):
            for m_id, m_data in data["models"].items():
                existing_models[m_id] = m_data
                if "full_id" in m_data:
                    existing_models[m_data["full_id"]] = m_data

        # Cas 2 : Format V1 existant (groupé sous 'tasks')
        elif "tasks" in data and isinstance(data["tasks"], dict):
            for task_id, models_list in data["tasks"].items():
                for m in models_list:
                    if not isinstance(m, dict):
                        continue
                    short_id = m.get("id") or m.get("full_id", "").replace("models/", "")
                    full_id = m.get("full_id") or f"models/{short_id}"

                    raw_in_mods = m.get("input_modalities", ["text"])
                    clean_in_mods = [x for x in raw_in_mods if x.lower() != "pdf"]
                    has_pdf = "pdf" in [x.lower() for x in raw_in_mods]

                    tier = m.get("tier", "standard")
                    if short_id in FEATURED_MODELS or m.get("is_featured"):
                        tier = "featured"

                    features = m.get("features")
                    if not features:
                        features = infer_model_features(
                            short_id,
                            m.get("supported_actions", []),
                            m.get("supports_thinking", False),
                        )

                    card = {
                        "id": short_id,
                        "full_id": full_id,
                        "display_name": m.get("display_name", short_id),
                        "family": "gemma" if "gemma" in short_id.lower() else "gemini",
                        "category": m.get("category", infer_category(short_id)),
                        "tier": tier,
                        "is_recommended": m.get("is_recommended", False),
                        "is_open_source": m.get("is_open_source", "gemma" in short_id.lower()),
                        "description": m.get("description", {}),
                        "tasks": [task_id] if isinstance(task_id, str) else ["text-generation"],
                        "modalities": {
                            "input": clean_in_mods,
                            "output": m.get("output_modalities", ["text"]),
                            "supports_documents": has_pdf,
                        },
                        "limits": {
                            "context_window": m.get("context_window"),
                            "max_output_tokens": m.get("max_output_tokens"),
                            "default_temperature": m.get("temperature", 1.0),
                        },
                        "features": features,
                        "supported_actions": m.get("supported_actions", []),
                        "rank": m.get("rank", 999),
                    }
                    existing_models[short_id] = card
                    existing_models[full_id] = card

        print(f"📖 Chargement réussi du registre précédent ({len(existing_models)} entrées indexées).")
    except Exception as e:
        print(f"⚠️ Impossible de charger l'ancien registre ({e}). Création d'une nouvelle base.")

    return existing_models, quick_picks


def build_v2_model_card(live_model, existing_card: dict = None) -> dict:
    """Construit une fiche V2 pour un modèle en fusionnant live API et annotations existantes."""
    full_id = live_model.name
    short_id = live_model.name.replace("models/", "")
    display_name = live_model.display_name or short_id

    # Actions et limites fournies par l'API
    supported_actions = list(getattr(live_model, "supported_actions", []) or [])
    context_window = getattr(live_model, "input_token_limit", None)
    max_output_tokens = getattr(live_model, "output_token_limit", None)
    temperature = getattr(live_model, "temperature", 1.0)
    supports_thinking = getattr(live_model, "thinking", None) is not None

    # Capacités fonctionnelles (features)
    features = infer_model_features(short_id, supported_actions, supports_thinking)

    # Modalités et tâche déduites
    sugg_task, in_mods, out_mods, supports_docs = detect_modalities_and_task_hint(
        short_id, getattr(live_model, "description", ""), supported_actions
    )

    # Récupération / Fusion des données humaines déjà annotées
    if existing_card:
        # Préservation stricte de la description existante (ex: bilingue FR/EN)
        desc = existing_card.get("description")
        if not desc or desc == "":
            desc = {"en": live_model.description} if live_model.description else {}

        tasks = existing_card.get("tasks", [sugg_task])
        if not tasks:
            tasks = [sugg_task]
        category = existing_card.get("category", infer_category(short_id))

        if short_id in FEATURED_MODELS:
            tier = "featured"
        else:
            tier = existing_card.get("tier", infer_tier(short_id))

        is_open_source = existing_card.get("is_open_source", "gemma" in short_id.lower())

        ex_modalities = existing_card.get("modalities", {})
        modalities = {
            "input": ex_modalities.get("input", in_mods),
            "output": ex_modalities.get("output", out_mods),
            "supports_documents": ex_modalities.get("supports_documents", supports_docs),
        }
    else:
        desc = {"en": live_model.description} if getattr(live_model, "description", None) else {}
        tasks = [sugg_task]
        category = infer_category(short_id)
        tier = infer_tier(short_id)
        is_open_source = "gemma" in short_id.lower()
        modalities = {
            "input": in_mods,
            "output": out_mods,
            "supports_documents": supports_docs,
        }

    family = "gemma" if "gemma" in short_id.lower() else "gemini"

    return {
        "id": short_id,
        "full_id": full_id,
        "display_name": display_name,
        "family": family,
        "category": category,
        "tier": tier,
        "is_recommended": False,
        "is_open_source": is_open_source,
        "description": desc,
        "tasks": tasks,
        "modalities": modalities,
        "limits": {
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "default_temperature": temperature if temperature is not None else 1.0,
        },
        "features": features,
        "supported_actions": supported_actions,
        "rank": 999,
    }


def main():
    parser = argparse.ArgumentParser(description="Mettre à jour le catalogue de modèles Google AI Studio (V2).")
    parser.add_argument("--dry-run", action="store_true", help="Simule l'exécution sans sauvegarder.")
    parser.add_argument("--force-offline", action="store_true", help="Recalcule le catalogue uniquement depuis le JSON local existant sans appeler l'API.")
    args = parser.parse_args()

    json_path = Path(__file__).resolve().parent / "google_ai_studio.json"

    # 1. Charger les données existantes pour préserver descriptions, catégories et quick_picks
    existing_models, quick_picks = load_existing_registry(json_path)

    # 2. Récupérer les modèles depuis l'API Google GenAI
    live_models = []
    if not args.force_offline:
        api_key = API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ Aucune clé GEMINI_API_KEY trouvée dans l'environnement ni dans le .env.")
            api_key = input("👉 Entrez votre clé API Gemini (ou appuyez sur Entrée pour mode local) : ").strip()

        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                print("📡 Récupération de la liste en direct depuis Google AI Studio...")
                live_models = list(client.models.list())
                print(f"✅ {len(live_models)} modèles récupérés depuis l'API.")
            except Exception as e:
                print(f"❌ Erreur lors de l'appel API Google AI Studio : {e}")
                print("Basculement sur les modèles du catalogue local existant...")

    # Si hors-ligne ou échec API, on se base sur les modèles déjà connus
    processed_models = {}

    if live_models:
        for lm in live_models:
            short_id = lm.name.replace("models/", "")
            ex_card = existing_models.get(short_id) or existing_models.get(lm.name)
            card = build_v2_model_card(lm, ex_card)
            processed_models[short_id] = card
    else:
        for m_id, m_data in existing_models.items():
            if m_id.startswith("models/"):
                continue
            processed_models[m_id] = m_data

    if not processed_models:
        print("❌ Aucun modèle disponible à traiter. Abandon.")
        return

    # 3. Recalculer le scoring et le classement (Ranking)
    sorted_cards = sorted(processed_models.values(), key=score_model_for_ranking, reverse=True)

    # Attribuer les rangs globaux et déterminer les recommandations par tâche
    task_leaders = {}
    for rank, card in enumerate(sorted_cards, 1):
        card["rank"] = rank
        primary_task = card["tasks"][0] if card["tasks"] else "text-generation"
        if primary_task not in task_leaders and card.get("tier") in ("featured", "standard"):
            task_leaders[primary_task] = card["id"]

    for card in sorted_cards:
        primary_task = card["tasks"][0] if card["tasks"] else "text-generation"
        card["is_recommended"] = (task_leaders.get(primary_task) == card["id"])

    # 4. Calcul dynamique du résumé des tâches (sans duplication de modèles)
    tasks_summary = {}
    for card in sorted_cards:
        for t in card.get("tasks", []):
            tasks_summary[t] = tasks_summary.get(t, 0) + 1

    # Dictionnaire ordonné des modèles
    final_models = {card["id"]: card for card in sorted_cards}

    # 5. Construction du payload final V2
    output_data = {
        "schema_version": "2.0.0",
        "provider": "google_ai_studio",
        "name": "Google AI Studio (Gemini)",
        "updated_at": datetime.now().isoformat(),
        "total_models": len(final_models),
        "tasks_summary": tasks_summary,
        "quick_picks": quick_picks,
        "models": final_models,
    }

    if args.dry_run:
        print("\n🔎 MODE SIMULATION (--dry-run) : Aucun fichier modifié.")
        sample_id = "gemini-3.8-flash" if "gemini-3.8-flash" in final_models else next(iter(final_models))
        print(f"Exemple de fiche pour '{sample_id}' :\n{json.dumps(final_models.get(sample_id), indent=2, ensure_ascii=False)}")
        return

    # Sauvegarde atomique du nouveau JSON
    temp_path = json_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    temp_path.replace(json_path)

    print("\n" + "=" * 70)
    print(" 🎉 REGISTRE V2.0.0 MIS À JOUR ET ENREGISTRÉ AVEC SUCCÈS")
    print("=" * 70)
    print(f"📊 Modèles totaux indexés : {len(final_models)}")
    print(f"📁 Fichier JSON cible     : {json_path}")
    print("\nRésumé dynamique des capacités par tâche :")
    for t_id, count in sorted(tasks_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {TASKS_MAP.get(t_id, t_id)} (`{t_id}`) : {count} modèle(s)")
    print("=" * 70)


if __name__ == "__main__":
    main()