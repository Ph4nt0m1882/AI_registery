# 🌌 AI Registry

Un registre centralisé, normalisé et open source des modèles d'IA (Google AI Studio, OpenAI, Anthropic, etc.), structuré par **tâches fonctionnelles** et **capacités multimodales**.

Ce projet permet à n'importe quelle application (comme [Muninn](https://github.com/Ph4nt0m1882/muninn), des orchestrateurs multi-agents ou des outils CLI) de consommer dynamiquement la liste à jour des modèles d'IA sans recompiler l'application.

---

## 📡 Accès direct aux catalogues JSON (Consommation API / App)

Vous pouvez consommer directement les fichiers JSON bruts dans vos projets :

### 1. Google AI Studio (Gemini, Imagen, Veo, Lyria...)
* **GitHub Raw** :  
  `https://raw.githubusercontent.com/Ph4nt0m1882/AI_registery/main/Google/google_ai_studio.json`
* **CDN jsDelivr (Ultra-rapide, mondial, sans rate-limit)** :  
  `https://cdn.jsdelivr.net/gh/Ph4nt0m1882/AI_registery@main/Google/google_ai_studio.json`

---

## 📂 Fournisseurs supportés

| Fournisseur | Statut | Modèles recensés | Tâches couvertes |
| :--- | :---: | :---: | :--- |
| **Google AI Studio** | 🟢 Actif | 53 modèles | Texte, Image, Vidéo (Veo), Musique (Lyria), TTS, Transcription, Live, Deep Research, Robotique, Embeddings |
| **Anthropic (Claude)** | 🟡 Bientôt | - | Text generation, Vision, Thinking |
| **OpenAI (GPT / DALL-E)**| 🟡 Bientôt | - | Text generation, Image, Speech, Embeddings |
| **Local (Candle / Ollama)**| 🟡 Bientôt | - | Modèles embarqués hors-ligne |

---

## 🧩 Structure du Schéma JSON

Chaque catalogue suit un schéma prédictible conçu pour les orchestrateurs multi-fournisseurs :

```json
{
  "schema_version": "1.0.0",
  "provider": "google_ai_studio",
  "updated_at": "2026-08-29T23:04:00",
  "total_models": 53,
  "tasks_summary": {
    "text-generation": 18,
    "text-to-image": 7,
    "text-to-video": 5,
    "text-to-music": 2,
    "text-to-speech": 3,
    "speech-to-text": 1,
    "realtime-multimodal": 6,
    "deep-research": 5,
    "robotics": 3,
    "embeddings": 3
  },
  "tasks": {
    "text-generation": [
      {
        "id": "gemini-2.5-flash",
        "full_id": "models/gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash",
        "description": "Stable version of Gemini 2.5 Flash, our mid-size multimodal model...",
        "task": "text-generation",
        "is_multimodal": true,
        "input_modalities": ["text", "image", "audio", "video", "pdf"],
        "output_modalities": ["text"],
        "context_window": 1048576,
        "max_output_tokens": 65536,
        "supports_thinking": true,
        "supported_actions": ["generateContent", "countTokens", "createCachedContent", "batchGenerateContent"]
      }
    ]
  }
}
```

---

## 🛠️ Mise à jour des catalogues

Pour rafraîchir la liste des modèles ou intégrer les nouveaux modèles sortis :

### 1. Prérequis
Installez [`uv`](https://github.com/astral-sh/uv) :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Configuration
Créez votre fichier `.env` à la racine :
```bash
cp .env.example .env
# Renseignez votre GEMINI_API_KEY dans .env
```

### 3. Exécution
* **Vérifier et trier les nouveaux modèles uniquement** :
  ```bash
  uv run Google/getter_gg_ai_studio.py
  ```
* **Tout reclasser interactivement à neuf** :
  ```bash
  uv run Google/getter_gg_ai_studio.py --reset
  ```

---

## 📄 Licence

Ce projet est distribué sous licence MIT. Libre à vous de l'intégrer dans vos applications commerciales ou open source.
