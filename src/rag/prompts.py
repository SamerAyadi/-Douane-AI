from src.rag.language import detect_question_language


LANGUAGE_INSTRUCTIONS = {
    "fr": """
Règles obligatoires :
- Réponds uniquement en français correct, clair et professionnel.
- Donne directement la réponse finale, sans raisonnement ni commentaire sur la tâche.
- Retourne uniquement du texte brut, jamais du JSON, du code ou des balises.
- Utilise seulement les faits explicitement présents dans les extraits.
- Ne mélange pas les règles provenant de documents ou de sections sans rapport.
- Pour une question oui/non ou une liste de noms, vérifie séparément chaque élément demandé.
- Conserve fidèlement les noms propres, numéros, dates, codes et références juridiques.
- Si l'OCR rend un élément illisible, ne le devine pas.
- Résume uniquement ce qui répond à la question et ne copie pas un long extrait.
- Si la source est arabe, rends son sens en français naturel sans traduction littérale.
- Si aucun extrait ne permet de répondre, écris uniquement : "L'information n'est pas disponible dans les documents fournis."
- Dans ce cas, n'ajoute ni explication ni source ; le système gère les citations.
- N'invente ni le fichier ni la page : le système ajoutera la source automatiquement.
- Rédige au maximum 110 mots.
""".strip(),
    "ar": """
قواعد إلزامية:
- أجب بالعربية الفصحى الواضحة والسليمة وبأسلوب مهني موجز.
- قدّم الإجابة النهائية مباشرة، من دون تحليل أو شرح للتفكير أو تعليق على المهمة.
- أعد نصًا عاديًا فقط، ولا تُرجع JSON أو شفرة أو وسومًا.
- استخدم فقط الحقائق الواردة بوضوح في المقتطفات.
- لا تخلط بين أحكام وثائق أو أقسام لا تتعلق بالسؤال نفسه.
- في أسئلة نعم أو لا وقوائم الأسماء، تحقّق من كل عنصر مطلوب على حدة.
- حافظ بدقة على الأسماء والأرقام والتواريخ والرموز والمراجع القانونية.
- إذا جعل التعرف الضوئي عنصرًا غير مقروء، فلا تخمّنه.
- لخّص فقط ما يجيب عن السؤال، ولا تنسخ مقتطفًا طويلًا.
- إذا كان المصدر فرنسيًا، فانقل معناه إلى عربية فصحى طبيعية غير حرفية.
- إذا لم يسمح أي مقتطف بالإجابة، فاكتب فقط: "المعلومة غير متوفرة في الوثائق المقدمة."
- في هذه الحالة لا تضف شرحًا أو مصدرًا؛ فالنظام يتولى إضافة الاستشهادات.
- لا تخترع اسم الملف أو الصفحة؛ سيضيف النظام المصدر تلقائيًا.
- اكتب فقرة أو فقرتين قصيرتين ولا تتجاوز 110 كلمات.
""".strip(),
}

PROMPT_INTRO = {
    "fr": "Réponds uniquement à partir des extraits documentaires ci-dessous.",
    "ar": "أجب بالاعتماد فقط على مقتطفات الوثائق أدناه.",
}

QUESTION_LABEL = {
    "fr": "Question",
    "ar": "السؤال",
}

REPAIR_INSTRUCTIONS = {
    "fr": (
        "Relis tous les extraits et remplace la première réponse si elle était vide, "
        "trop longue, dans une mauvaise langue, sous forme JSON, ou si elle déclarait "
        "à tort que l'information était indisponible."
    ),
    "ar": (
        "أعد قراءة جميع المقتطفات واستبدل الإجابة الأولى إذا كانت فارغة أو طويلة أو "
        "بلغة خاطئة أو بصيغة JSON أو ادعت خطأً أن المعلومات غير متوفرة."
    ),
}


def build_rag_prompt(question: str, context: str) -> str:
    language = detect_question_language(question)
    return f"""
{PROMPT_INTRO[language]}

{context}

{QUESTION_LABEL[language]} : {question}

{LANGUAGE_INSTRUCTIONS[language]}
""".strip()


def build_repair_prompt(question: str, context: str) -> str:
    language = detect_question_language(question)
    return f"""
{PROMPT_INTRO[language]}

{context}

{QUESTION_LABEL[language]} : {question}

{REPAIR_INSTRUCTIONS[language]}

{LANGUAGE_INSTRUCTIONS[language]}
""".strip()