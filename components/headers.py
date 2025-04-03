def get_main_title():
    return """
    <h1> 🧠 Redes Neuronales para Predecir la Dinámica de un Balín</h1>
    """

def get_intro_highlight():
    return """
    <div class="justified-text highlight">
    Este dashboard presenta el diseño, implementación y evaluación de modelos de deep learning 
    para predecir la trayectoria de un balín bajo un campo magnético armónico.  
    La estructura y metodologías se basan en el documento del examen de desarrollo de proyectos.
    </div>
    """

def get_section_header(num, icon, title):
    return f"""
    <div class="section-header"><h2>{icon} {num}. {title}</h2></div>
    """