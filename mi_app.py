
import gradio as gr
import pandas as pd
import joblib

# Carga el modelo de predicción
modelo = joblib.load("xgboost_abandono_escolar_modelo.pkl")

# Función para predecir el riesgo
def predecir_riesgo(
    edad, genero, convive, hijos, nivel_madre, nivel_padre,
    trabaja, trabajo_mas_30, ingresos_trabajo, necesita_trabajar, trabajo_campo,
    escuela_secundaria, promedio_sec, prom_lenmat, aprobo_ingreso,
    est_regular, quedo_libre, materias_reg_2024, materias_adeudafinal,
    acceso_internet, dispositivos_estudio, recibe_beca, boleto_estudiantil,
    acceso_tutorias, apoyo_psico, motivacion_carr, concentracion,
    bullying, ansiedad_ang, angust_expo, vergue_expo,
    leer_cantidad, llegar_univ, cansa_llegar, transporte,
    mov_propia, recreativas, realiz_recreat, enfermedad,
    gust_carrera, exp_terminar, cambiar_carre, prov_undec,
    primera_elecc, horario_adec, apoyo_compa, relacion_docentes,
    adaptacion, futuro_exp
):
    # Organiza todos los inputs en un diccionario
    datos = {
        "edad": edad, "genero": genero, "convive": convive, "hijos": hijos,
        "nivel_madre": nivel_madre, "nivel_padre": nivel_padre, "trabaja": trabaja,
        "trabajo_mas_30": trabajo_mas_30, "ingresos_trabajo": ingresos_trabajo,
        "necesita_trabajar": necesita_trabajar, "trabajo_campo": trabajo_campo,
        "escuela_secundaria": escuela_secundaria, "promedio_sec": promedio_sec,
        "prom_lenmat": prom_lenmat, "aprobo_ingreso": aprobo_ingreso,
        "est_regular": est_regular, "quedo_libre": quedo_libre,
        "materias_reg_2024": materias_reg_2024, "materias_adeudafinal": materias_adeudafinal,
        "acceso_internet": acceso_internet, "dispositivos_estudio": dispositivos_estudio,
        "recibe_beca": recibe_beca, "boleto_estudiantil": boleto_estudiantil,
        "acceso_tutorias": acceso_tutorias, "apoyo_psico": apoyo_psico,
        "motivacion_carr": motivacion_carr, "concentracion": concentracion,
        "bullying": bullying, "ansiedad_ang": ansiedad_ang,
        "angust_expo": angust_expo, "vergue_expo": vergue_expo,
        "leer_cantidad": leer_cantidad, "llegar_univ": llegar_univ,
        "cansa_llegar": cansa_llegar, "transporte": transporte,
        "mov_propia": mov_propia, "recreativas": recreativas,
        "realiz_recreat": realiz_recreat, "enfermedad": enfermedad,
        "gust_carrera": gust_carrera, "exp_terminar": exp_terminar,
        "cambiar_carre": cambiar_carre, "prov_undec": prov_undec,
        "primera_elecc": primera_elecc, "horario_adec": horario_adec,
        "apoyo_compa": apoyo_compa, "relacion_docentes": relacion_docentes,
        "adaptacion": adaptacion, "futuro_exp": futuro_exp
    }

    df = pd.DataFrame([datos])
    prob = modelo.predict_proba(df)[0, 1]
    riesgo = "ALTO" if prob >= 0.4 else "BAJO"
    return f"🔍 Riesgo de abandono: {riesgo} ({prob:.2%})"

with gr.Blocks() as app:
    # Logo
    logo = gr.Image(value="Logu.png", label="Universidad", show_label=False, interactive=False)
    gr.Markdown("## 🎓 Sistema de Alerta Temprana del Abandono Universitario")
    
    # Sección de pestañas
    with gr.Tab("🧍 Datos Personales"):
        edad = gr.Number(label="Edad")
        genero = gr.Radio(choices=[0, 1], label="Género (0=mujer, 1=varón)")
        convive = gr.Radio(choices=[0, 1], label="¿Convive con alguien?")
        hijos = gr.Number(label="Cantidad de hijos")
        nivel_madre = gr.Slider(0, 3, step=1, label="Nivel educativo madre")
        nivel_padre = gr.Slider(0, 3, step=1, label="Nivel educativo padre")

    with gr.Tab("💼 Trabajo e ingresos"):
        trabaja = gr.Radio([0, 1], label="¿Trabaja?")
        trabajo_mas_30 = gr.Radio([0, 1], label="¿Trabaja más de 30hs?")
        ingresos_trabajo = gr.Radio([0, 1], label="¿Tiene ingresos del trabajo?")
        necesita_trabajar = gr.Radio([0, 1], label="¿Necesita trabajar?")
        trabajo_campo = gr.Radio([0, 1], label="¿Trabaja en el campo?")

    with gr.Tab("🏫 Historia Escolar"):
        escuela_secundaria = gr.Radio([0, 1], label="¿Completó secundaria?")
        promedio_sec = gr.Number(label="Promedio de secundaria")
        prom_lenmat = gr.Number(label="Promedio lengua y matemática")
        aprobo_ingreso = gr.Radio([0, 1], label="¿Aprobó el ingreso?")
        est_regular = gr.Radio([0, 1], label="¿Es estudiante regular?")
        quedo_libre = gr.Radio([0, 1], label="¿Quedó libre alguna vez?")
        materias_reg_2024 = gr.Number(label="Materias regulares 2024")
        materias_adeudafinal = gr.Number(label="Finales adeudados")

    with gr.Tab("📶 Recursos y apoyos"):
        acceso_internet = gr.Radio([0, 1], label="¿Tiene internet?")
        dispositivos_estudio = gr.Number(label="Cantidad de dispositivos")
        recibe_beca = gr.Radio([0, 1], label="¿Recibe beca?")
        boleto_estudiantil = gr.Radio([0, 1], label="¿Tiene boleto?")
        acceso_tutorias = gr.Radio([0, 1], label="¿Accede a tutorías?")
        apoyo_psico = gr.Radio([0, 1], label="¿Accede a apoyo psicológico?")

    with gr.Tab("🧠 Bienestar y emociones"):
        motivacion_carr = gr.Slider(0, 3, step=1, label="Motivación con la carrera")
        concentracion = gr.Slider(0, 3, step=1, label="Nivel de concentración")
        bullying = gr.Radio([0, 1], label="¿Sufre bullying?")
        ansiedad_ang = gr.Slider(0, 3, step=1, label="Ansiedad")
        angust_expo = gr.Slider(0, 3, step=1, label="Angustia ante exposición")
        vergue_expo = gr.Slider(0, 3, step=1, label="Vergüenza en exposiciones")

    with gr.Tab("🚗 Accesibilidad"):
        leer_cantidad = gr.Number(label="Libros que leyó el año pasado")
        llegar_univ = gr.Radio([0, 1], label="¿Le cuesta llegar a la universidad?")
        cansa_llegar = gr.Radio([0, 1], label="¿Se cansa al llegar?")
        transporte = gr.Radio([0, 1], label="¿Usa transporte público?")
        mov_propia = gr.Radio([0, 1], label="¿Tiene movilidad propia?")

    with gr.Tab("🎯 Vocación y entorno"):
        recreativas = gr.Radio([0, 1], label="¿Participa en actividades recreativas?")
        realiz_recreat = gr.Radio([0, 1], label="¿Realiza actividades culturales/deportivas?")
        enfermedad = gr.Radio([0, 1], label="¿Tiene enfermedades?")
        gust_carrera = gr.Slider(0, 3, step=1, label="¿Le gusta la carrera?")
        exp_terminar = gr.Slider(0, 3, step=1, label="¿Espera terminar?")
        cambiar_carre = gr.Radio([0, 1], label="¿Quiere cambiar de carrera?")
        prov_undec = gr.Radio([0, 1], label="¿Viene de otra provincia?")
        primera_elecc = gr.Radio([0, 1], label="¿Fue su primera elección?")
        horario_adec = gr.Radio([0, 1], label="¿Le sirve el horario?")
        apoyo_compa = gr.Radio([0, 1], label="¿Tiene apoyo de compañeros?")
        relacion_docentes = gr.Radio([0, 1], label="¿Relación con docentes?")
        adaptacion = gr.Slider(0, 3, step=1, label="¿Se adapta a nuevos entornos?")
        futuro_exp = gr.Slider(0, 3, step=1, label="¿Ve un buen futuro profesional?")

    # Botón de predicción
    pred_btn = gr.Button("Predecir riesgo")
    output = gr.Textbox(label="Resultado")
    pred_btn.click(predecir_riesgo, inputs=[edad, genero, convive, hijos, nivel_madre, nivel_padre,
                                           trabaja, trabajo_mas_30, ingresos_trabajo, necesita_trabajar,
                                           trabajo_campo, escuela_secundaria, promedio_sec, prom_lenmat,
                                           aprobo_ingreso, est_regular, quedo_libre, materias_reg_2024,
                                           materias_adeudafinal, acceso_internet, dispositivos_estudio,
                                           recibe_beca, boleto_estudiantil, acceso_tutorias, apoyo_psico,
                                           motivacion_carr, concentracion, bullying, ansiedad_ang,
                                           angust_expo, vergue_expo, leer_cantidad, llegar_univ,
                                           cansa_llegar, transporte, mov_propia, recreativas,
                                           realiz_recreat, enfermedad, gust_carrera, exp_terminar,
                                           cambiar_carre, prov_undec, primera_elecc, horario_adec,
                                           apoyo_compa, relacion_docentes, adaptacion, futuro_exp],
                      outputs=output)

app.launch()
