# Résumé technique d'avancement — Douane-AI

## Objectif

Douane-AI est un assistant documentaire bilingue arabe/français pour interroger les textes officiels de la Douane tunisienne. La solution fonctionne localement afin de garder le contrôle sur les documents et les traitements.

## Réalisations

Le projet couvre aujourd'hui toute la chaîne fonctionnelle. Un scraper récupère les PDF officiels et leurs métadonnées. L'ingestion tente d'abord PyMuPDF et pdfplumber, mesure la qualité du texte, puis applique Tesseract en arabe et en français avec plusieurs modes de segmentation pour les pages scannées. Le meilleur texte est nettoyé, découpé et indexé dans ChromaDB avec des embeddings multilingues.

Le retriever sélectionne les passages pertinents en tenant compte de la langue et des numéros de documents. Qwen3, exécuté localement avec Ollama, génère une réponse courte en arabe ou en français et cite la source. Lorsque l'information n'est pas présente, le système renvoie une réponse prudente sans fausse citation.

FastAPI expose le chatbot et PostgreSQL conserve les sessions et les messages. Une réécriture déterministe rend les questions de suivi comme « اشرح ذلك أكثر » autonomes avant la recherche, tout en sauvegardant la question originale. L'interface Angular propose l'historique des conversations, le chargement d'une session, le RTL/LTR et l'affichage propre des sources.

## Résultats mesurés

La dernière évaluation de 18 questions obtient 16/16 documents corrects et 16/16 pages correctes dans le top 3 pour les questions répondables. La langue est correcte pour 18/18 réponses, les deux questions absentes sont traitées correctement et le score global des contrôles est de 94 %. Le temps moyen complet est d'environ 17 secondes sur la machine locale.

## Travaux restants

Les prochaines étapes sont d'élargir progressivement le corpus, de refaire les évaluations après chaque ajout, d'améliorer les cas OCR tabulaires encore difficiles, puis de finaliser les tests d'intégration et le rapport. L'authentification, les rôles, l'administration des documents et le déploiement peuvent être ajoutés ensuite sans modifier le cœur RAG.
