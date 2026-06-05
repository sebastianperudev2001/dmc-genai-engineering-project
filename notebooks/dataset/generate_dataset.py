import json

PROGRAMS = [
    "power-bi-especializacion", "diploma-data-analyst",
    "machine-learning-especializacion", "diploma-data-scientist",
    "azure-data-engineering", "diploma-data-engineer",
    "n8n-chatbots-especializacion", "diseno-chatbots-python",
    "ia-ejecutivos", "people-analytics",
    "membresia-datapro", "membresia-ia-developer",
]

# 10 ejemplos base por motivación — se ciclan hasta llegar a 40 cada una
SAMPLES = {
    "salary": [
        ("Quiero aprender Power BI, mi jefe me dijo que si lo manejo bien me suben el sueldo.", "warm", "power-bi-especializacion"),
        ("¿Cuánto gana un data scientist en Perú? Quiero cambiar de área para ganar más.", "warm", "diploma-data-scientist"),
        ("Estoy como analista financiero y quiero aprender Python para que me paguen mejor.", "warm", "diploma-data-analyst"),
        ("Mi colega que estudió aquí consiguió un aumento. ¿Qué curso me recomiendas?", "warm", "membresia-datapro"),
        ("Quiero pasar de 3000 a 5000 soles, ¿es posible con sus cursos?", "warm", "diploma-data-scientist"),
        ("¿Cuánto tiempo tarda en verse el retorno de la inversión de sus programas?", "warm", "diploma-data-analyst"),
        ("Trabajo en marketing y quiero aprender analytics para pedir aumento.", "warm", "power-bi-especializacion"),
        ("¿Tienen testimonios de gente que haya conseguido trabajo mejor pagado?", "warm", "membresia-datapro"),
        ("Soy contador y quiero pasar a datos para ganar el doble.", "warm", "diploma-data-analyst"),
        ("¿El diploma de data science mejora el perfil para negociar sueldo?", "warm", "diploma-data-scientist"),
    ],
    "growth": [
        ("Quiero cambiar de carrera, actualmente soy contador y quiero ser data analyst.", "warm", "diploma-data-analyst"),
        ("Estoy en recursos humanos y quiero especializarme en people analytics.", "warm", "people-analytics"),
        ("Quiero entrar al mundo de la inteligencia artificial, ¿por dónde empiezo?", "warm", "membresia-ia-developer"),
        ("Tengo 5 años en finanzas y quiero agregar habilidades de datos a mi perfil.", "warm", "diploma-data-analyst"),
        ("Quiero ser data engineer, ¿qué camino me recomiendan?", "hot", "diploma-data-engineer"),
        ("Me gustaría especializarme en machine learning para trabajar en una startup.", "warm", "diploma-data-scientist"),
        ("Soy QA y quiero migrar a data science, ¿tienen algo para mi perfil?", "warm", "diploma-data-scientist"),
        ("¿Cuál es el programa que más me abre puertas en el mercado laboral peruano?", "warm", "diploma-data-scientist"),
        ("Soy recién egresado y quiero especializarme en datos desde cero.", "warm", "diploma-data-analyst"),
        ("Quiero hacer una transición de administración a inteligencia artificial.", "warm", "membresia-ia-developer"),
    ],
    "company_requirement": [
        ("Mi empresa me pidió que me certifique en Azure, es un requisito para el área.", "hot", "azure-data-engineering"),
        ("El área de TI de donde trabajo necesita que todos aprendamos Power BI.", "hot", "power-bi-especializacion"),
        ("Mi jefe me dijo que si no aprendo Python me cambian de área.", "hot", "diploma-data-analyst"),
        ("La empresa donde trabajo quiere implementar IA y necesito liderarla.", "hot", "ia-ejecutivos"),
        ("¿Tienen facturación a empresa? Lo pregunto porque mi empleador pagaría.", "hot", "diploma-data-engineer"),
        ("Necesito aprender N8N para automatizar procesos en mi trabajo.", "hot", "n8n-chatbots-especializacion"),
        ("El área de RRHH de mi empresa quiere hacer analytics de personas.", "hot", "people-analytics"),
        ("Me pidieron desarrollar chatbots para el área de soporte.", "hot", "diseno-chatbots-python"),
        ("Tengo plazo de 2 meses para certificarme en datos, es un requisito.", "hot", "diploma-data-analyst"),
        ("Mi empresa exige que el área de finanzas maneje Power BI.", "hot", "power-bi-especializacion"),
    ],
    "academic": [
        ("Estoy estudiando ingeniería y quiero complementar con data science.", "cold", "diploma-data-scientist"),
        ("Me interesa la IA por curiosidad, quiero entender cómo funciona.", "cold", "ia-ejecutivos"),
        ("Vi un video sobre machine learning y me pareció interesante.", "cold", "membresia-datapro"),
        ("Quiero aprender Python aunque sea lo básico.", "cold", "diploma-data-analyst"),
        ("¿Tienen cursos para alguien que recién empieza?", "cold", "membresia-datapro"),
        ("Soy universitario y quiero aprender algo de datos en mis tiempos libres.", "cold", "membresia-datapro"),
        ("¿Cuál es la diferencia entre data analyst y data scientist?", "cold", "diploma-data-analyst"),
        ("Me gusta la estadística y quiero ver si tiene aplicación práctica.", "cold", "diploma-data-scientist"),
        ("Estoy en mi último año de universidad y quiero especializarme.", "cold", "diploma-data-analyst"),
        ("Quiero entender qué es el machine learning sin tener experiencia previa.", "cold", "membresia-ia-developer"),
    ],
    "undefined": [
        ("Hola, ¿qué cursos tienen?", "cold", "membresia-datapro"),
        ("Vi su publicidad en Instagram.", "cold", "membresia-datapro"),
        ("Un amigo me recomendó este lugar.", "cold", "membresia-datapro"),
        ("¿Cuánto cuestan sus programas?", "warm", "diploma-data-analyst"),
        ("Quiero algo relacionado con tecnología.", "cold", "membresia-ia-developer"),
        ("¿Tienen modalidad virtual?", "cold", "membresia-datapro"),
        ("Necesito un certificado.", "warm", "diploma-data-analyst"),
        ("¿Cuánto dura el diploma de data science?", "warm", "diploma-data-scientist"),
        ("Me recomendaron que aprenda datos.", "cold", "membresia-datapro"),
        ("¿Cuál es el mejor curso para empezar?", "cold", "membresia-datapro"),
    ],
}

SYSTEM_PROMPT = (
    "Eres un clasificador de leads para DMC Institute. "
    "Dado un mensaje de usuario, responde SOLO con JSON: "
    '{"motivation": "growth|salary|company_requirement|academic|undefined", '
    '"score": "hot|warm|cold", "recommended_program": "<slug>", "justification": "<1 oración>"}'
)

def build_dataset() -> dict:
    conversations = []
    conv_id = 1
    for motivation_key, samples in SAMPLES.items():
        # ciclar para llegar a 40 por categoría
        extended = (samples * 4)[:40]
        for user_msg, score, program in extended:
            assistant_output = json.dumps({
                "motivation": motivation_key,
                "score": score,
                "recommended_program": program,
                "justification": f"Motivación {motivation_key} detectada en el mensaje.",
            }, ensure_ascii=False)
            conversations.append({
                "id": f"lead_{conv_id:03d}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_output},
                ],
            })
            conv_id += 1
    return {"conversations": conversations}

if __name__ == "__main__":
    dataset = build_dataset()
    with open("dmc_leads.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"✅ Dataset generado: {len(dataset['conversations'])} ejemplos")
    from collections import Counter
    motivations = [
        json.loads(c["messages"][2]["content"])["motivation"]
        for c in dataset["conversations"]
    ]
    print(Counter(motivations))
