# Checklist de test — Douane-AI

## Préparation

- [ ] PostgreSQL est démarré et `DATABASE_URL` est valide.
- [ ] Ollama est démarré et `qwen3:4b` est disponible.
- [ ] ChromaDB contient le corpus attendu.
- [ ] FastAPI répond sur `http://127.0.0.1:8000/health`.
- [ ] Angular répond sur `http://127.0.0.1:4200`.

## Données et OCR

- [ ] Le scraper télécharge un petit lot sans doublons.
- [ ] Les URL et métadonnées sont enregistrées.
- [ ] Un PDF numérique utilise une extraction native correcte.
- [ ] Un PDF scanné déclenche l'OCR `ara+fra`.
- [ ] Une page difficile compare PSM 3, 6 et 11.
- [ ] Les chunks conservent source, page, langue et méthode d'extraction.

## Recherche et génération

- [ ] Une question arabe récupère le bon document arabe lorsqu'il existe.
- [ ] Une question française récupère le bon document disponible.
- [ ] Une question par numéro retrouve le document demandé.
- [ ] La réponse utilise la langue de la question.
- [ ] La source affichée correspond au document récupéré.
- [ ] Une question absente renvoie le message d'indisponibilité sans source.
- [ ] Une liste ou un tableau conserve les codes et nombres importants.

## Conversations

- [ ] Le premier `POST /chat` crée un `session_id`.
- [ ] Le second message réutilise le même `session_id`.
- [ ] `اشرح ذلك أكثر` reste sur le sujet précédent.
- [ ] Une nouvelle question explicite sur 025/2026 n'est pas réécrite avec le contexte 026/2026.
- [ ] PostgreSQL stocke la question originale, pas la requête réécrite.
- [ ] `/sessions` retourne les sessions par date de mise à jour.
- [ ] `/sessions/{id}/messages` retourne les messages dans l'ordre.

## Interface Angular

- [ ] La liste des sessions se charge au démarrage.
- [ ] Cliquer une session recharge ses messages.
- [ ] « Nouveau » vide l'écran et démarre une nouvelle session au prochain envoi.
- [ ] Les messages arabes sont RTL et les messages français LTR.
- [ ] Les noms de PDF longs ne cassent pas la mise en page.
- [ ] Les réponses absentes n'affichent aucune carte source.
- [ ] Une panne de `/sessions` n'empêche pas d'utiliser le chat.
- [ ] L'interface reste lisible sur écran mobile.

## Régression automatisée

```powershell
python -m py_compile src\api\routes\chat.py src\api\services\question_rewriter.py
git diff --check
python -m evaluation.run_evaluation --mode retrieval
python -m evaluation.run_evaluation --mode chain
cd frontend
npm run build
```

## Validation PFE

- [ ] Les résultats d'évaluation sont archivés avec leur date.
- [ ] Les captures d'écran correspondent à la version démontrée.
- [ ] Les limites OCR et LLM sont expliquées dans le rapport.
- [ ] `.env`, `data/raw/`, `data/metadata/` et `chroma_db/` restent ignorés.
