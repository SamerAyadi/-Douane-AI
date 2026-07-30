from src.rag.language import detect_question_language


LANGUAGE_INSTRUCTIONS = {
    "fr": """
Règles obligatoires :
- Réponds uniquement en français correct, clair et professionnel.
- Donne directement la réponse finale, sans raisonnement, analyse ou commentaire sur la tâche.
- Retourne uniquement du texte brut, jamais du JSON, du code ou des balises.
- Résume seulement les informations utiles à la question. Ne copie pas un long extrait.
- Les titres, listes, procédures, conditions, pièces, délais et plafonds sont des informations
  valides et doivent être synthétisés lorsqu'ils répondent à la question.
- Si le contexte provient exceptionnellement d'une source arabe, rends son sens naturellement
  en français sans traduction littérale maladroite.
- Indique que l'information n'est pas disponible uniquement si aucun extrait ne traite du sujet.
- Le système ajoutera automatiquement le document source et la page : ne les invente pas.
- Rédige au maximum 110 mots.
""".strip(),
    "ar": """
قواعد إلزامية:
- أجب بالعربية الفصحى الواضحة والسليمة وبأسلوب مهني موجز.
- قدّم الإجابة النهائية مباشرة، من دون تحليل أو شرح للتفكير أو تعليق على المهمة.
- أعد نصًا عاديًا فقط، ولا تُرجع JSON أو شفرة أو وسومًا.
- لخّص المعلومات المفيدة للسؤال فقط، ولا تنسخ مقتطفًا طويلًا.
- اعتبر العناوين والقوائم والإجراءات والشروط والوثائق المطلوبة والآجال والحدود المالية
  معلومات صالحة يجب تلخيصها عندما تجيب عن السؤال.
- إذا كانت المقتطفات عربية، فاعتمد مصطلحاتها العربية مباشرة ولا تترجمها عن الفرنسية.
- إذا استُخدمت مقتطفات فرنسية احتياطيًا، فانقل معناها إلى عربية فصحى طبيعية غير حرفية،
  ولا تُخفِ اسم المصدر الفرنسي أو تستبدله بمصدر عربي.
- عند الاعتماد على مصدر فرنسي، استخدم مقابلات إدارية سليمة، منها:
  dépenses éligibles = المصاريف القابلة للتعويض،
  dépenses non éligibles = المصاريف غير القابلة للتعويض،
  loisirs personnels = الترفيه الشخصي،
  frais de réception = مصاريف استقبال العملاء والاجتماعات،
  approbation préalable = الموافقة المسبقة، conserver les reçus = الاحتفاظ بالإيصالات،
  formulaire de demande = استمارة طلب التعويض، traitement = معالجة الطلب،
  hébergement standard = الإقامة الفندقية العادية، transport routier = النقل البري.
  dîners officiels = مآدب العشاء الرسمية، Finance = قسم المالية،
  dépenses > 500 DT = المصاريف التي تتجاوز 500 دينار،
  assurance ou frais médicaux = التأمين أو المصاريف الطبية.
  fournitures et équipement de bureau = لوازم ومعدات المكتب،
  amendes de stationnement ou violations routières = غرامات الوقوف أو المخالفات المرورية،
  hébergement hôtel (standard/capitale) = الإقامة الفندقية (فندق عادي/العاصمة)،
  traitement dans 15 jours ouvrables = معالجة الطلب خلال 15 يوم عمل.
- اكتب فقرة أو فقرتين قصيرتين، وتجنب القوائم الطويلة والصياغات الركيكة.
- توقّف بعد عرض الحقائق المطلوبة، ولا تضف خاتمة عامة أو جملة حشو.
- اذكر أن المعلومات غير متوفرة فقط إذا لم يتناول أي مقتطف الموضوع المطلوب.
- لا تذكر اسم الملف أو الصفحة داخل متن الإجابة؛ سيضيف النظام سطر المصدر تلقائيًا.
- لا تتجاوز الإجابة 110 كلمات.
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