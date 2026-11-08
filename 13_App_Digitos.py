import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Dígitos · Daniel Flórez",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =====================================================================
#  AJUSTES
# =====================================================================
NOMBRES_MODELO = ("mnist_model.keras", "mnist_model.h5")  # se busca en este orden
GROSOR_TRAZO = 18        # grosor del lápiz en el lienzo (lienzo de 280 × 280)
MIN_TRAZO_PX = 14        # un dibujo más pequeño que esto se considera ruido
CONFIANZA_ALTA = 90
CONFIANZA_MEDIA = 60

# =====================================================================
#  ESTILO (paleta fría basada en #3f71a6)
# =====================================================================
PRIMARIO = "#3f71a6"
OSCURO = "#1b2f45"
CLARO = "#eaf2f9"
SUAVE = "#a9c7e3"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; font-size: 1.06rem; }
.stApp { background: linear-gradient(180deg, #f6fafd 0%, __CLARO__ 100%); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 860px; padding-top: 3rem; padding-bottom: 4rem; }

/* Hero */
.hero { text-align: center; padding: 1rem 0 .5rem 0; }
.hero .etiqueta {
    color: __PRIMARIO__; letter-spacing: .25em; text-transform: uppercase;
    font-size: .85rem; font-weight: 600;
}
.hero h1 {
    font-family: 'Playfair Display', serif; font-size: 3.1rem; color: __OSCURO__;
    margin: .3rem 0 .6rem 0; line-height: 1.12;
}
.hero p { color: #4d6479; font-size: 1.2rem; max-width: 620px; margin: 0 auto; }
.linea { width: 70px; height: 4px; border-radius: 4px; background: __PRIMARIO__; margin: 1.4rem auto; }
h2, h3 { font-family: 'Playfair Display', serif !important; color: __OSCURO__ !important; }
[data-testid="stHeaderActionElements"] { display: none !important; }

/* Tarjetas (contenedores con key="tarjeta_...") */
[class*="st-key-tarjeta"] {
    background: #ffffff !important;
    border: 1px solid __SUAVE__88 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(63, 113, 166, .08);
    padding: 1.4rem 1.5rem 1.3rem 1.5rem !important;
}
/* Texto legible dentro de las tarjetas, también si el navegador está en modo oscuro */
[class*="st-key-tarjeta"] p,
[class*="st-key-tarjeta"] label,
[class*="st-key-tarjeta"] li,
[class*="st-key-tarjeta"] [data-testid="stWidgetLabel"] * { color: __OSCURO__ !important; }
[class*="st-key-tarjeta"] [data-testid="stCaptionContainer"] * { color: #4d6479 !important; }

/* Botones e inputs */
.stButton > button {
    background: __PRIMARIO__; color: white; border: 1.5px solid __PRIMARIO__;
    border-radius: 999px; padding: .65rem 1.4rem; font-weight: 600; transition: all .2s ease;
}
.stButton > button:hover { background: __OSCURO__; border-color: __OSCURO__; color: white; transform: translateY(-2px); }
[data-testid="stFileUploaderDropzone"] { background: __CLARO__; border: 2px dashed __SUAVE__; border-radius: 16px; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stImage"] img { border-radius: 14px; }

/* Resultado */
.vacio {
    border: 2px dashed __SUAVE__; border-radius: 16px; padding: 2.4rem 1rem; text-align: center;
    color: #4d6479; background: #f6fafd; line-height: 1.5;
}
.resultado { text-align: center; }
.resultado .digito {
    font-family: 'Playfair Display', serif; font-size: 6.2rem; font-weight: 700; line-height: 1;
    color: __PRIMARIO__; margin: 0;
}
.resultado .confianza { font-size: 1.05rem; font-weight: 600; color: __OSCURO__; margin: .2rem 0 .1rem 0; }
.resultado .nota { font-size: .92rem; color: #4d6479; margin-bottom: .6rem; }
.barras { display: flex; align-items: flex-end; justify-content: center; gap: 6px; height: 96px; margin: .8rem 0 .2rem 0; }
.barra { flex: 1; max-width: 28px; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }
.barra .relleno { width: 100%; border-radius: 6px 6px 2px 2px; background: __SUAVE__; min-height: 3px; }
.barra.top .relleno { background: __PRIMARIO__; }
.etiquetas { display: flex; justify-content: center; gap: 6px; }
.etiquetas span { flex: 1; max-width: 28px; text-align: center; font-size: .82rem; color: #4d6479; }
.etiquetas span.top { font-weight: 700; color: __OSCURO__; }
.vista { display: flex; align-items: center; justify-content: center; gap: 14px; margin-top: 1rem; text-align: left; }
.vista img { width: 84px; height: 84px; image-rendering: pixelated; border-radius: 8px; border: 1px solid __SUAVE__; background: #000; }
.vista small { color: #4d6479; font-size: .85rem; line-height: 1.4; }

/* Sidebar */
[data-testid="stSidebar"] { background: __OSCURO__; }
[data-testid="stSidebar"] * { color: #e6f0f9 !important; }

#MainMenu, footer { visibility: hidden; }
</style>
"""


def inyectar_css() -> None:
    css = CSS
    for token, valor in {
        "__PRIMARIO__": PRIMARIO, "__OSCURO__": OSCURO, "__CLARO__": CLARO, "__SUAVE__": SUAVE,
    }.items():
        css = css.replace(token, valor)
    st.markdown(css, unsafe_allow_html=True)


# =====================================================================
#  LIENZO PARA DIBUJAR (componente propio: sin librerías externas)
# =====================================================================
LIENZO_HTML = """
<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: transparent; font-family: 'Inter', sans-serif; }
  .caja { width: 100%; max-width: 320px; margin: 0 auto; padding: 4px 2px 6px 2px; }
  .marco { position: relative; border-radius: 16px; overflow: hidden; border: 2px solid __SUAVE__;
           box-shadow: 0 8px 22px rgba(27, 47, 69, .18); background: #000; line-height: 0; }
  canvas { width: 100%; aspect-ratio: 1 / 1; display: block; touch-action: none; cursor: crosshair; }
  .pista { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
           color: rgba(255,255,255,.45); font-size: 1rem; line-height: 1.4; text-align: center;
           pointer-events: none; padding: 1rem; transition: opacity .2s; }
  .pista.oculta { opacity: 0; }
  .botones { display: flex; gap: 10px; margin-top: 12px; }
  button { flex: 1; padding: .6rem 0; border-radius: 999px; font: 600 .95rem 'Inter', sans-serif;
           cursor: pointer; border: 1.5px solid __PRIMARIO__; transition: all .2s ease; }
  #deshacer { background: transparent; color: __PRIMARIO__; }
  #deshacer:hover { background: __CLARO__; }
  #borrar { background: __PRIMARIO__; color: #fff; }
  #borrar:hover { background: __OSCURO__; border-color: __OSCURO__; }
</style></head>
<body>
<div class="caja">
  <div class="marco">
    <canvas id="lienzo" width="280" height="280"></canvas>
    <div class="pista" id="pista">Dibuja un dígito<br>del 0 al 9, grande y centrado</div>
  </div>
  <div class="botones">
    <button id="deshacer" type="button">Deshacer</button>
    <button id="borrar" type="button">Borrar todo</button>
  </div>
</div>
<script>
  const N = 280, GROSOR = __GROSOR__;
  const cv = document.getElementById("lienzo");
  const ctx = cv.getContext("2d");
  const pista = document.getElementById("pista");
  let trazos = [], actual = null, contador = 0;

  function mensaje(tipo, extra) {
    window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type: tipo }, extra), "*");
  }
  const listo = () => mensaje("streamlit:componentReady", { apiVersion: 1 });
  const altura = () => mensaje("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight });
  const enviar = () => {
    contador += 1;
    const img = trazos.length ? cv.toDataURL("image/png") : "";
    mensaje("streamlit:setComponentValue", { value: { n: contador, img: img }, dataType: "json" });
  };

  function dibujar() {
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, N, N);
    ctx.strokeStyle = "#fff"; ctx.fillStyle = "#fff";
    ctx.lineWidth = GROSOR; ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const t of trazos) {
      if (t.length === 1) {
        ctx.beginPath(); ctx.arc(t[0][0], t[0][1], GROSOR / 2, 0, Math.PI * 2); ctx.fill();
      } else {
        ctx.beginPath(); ctx.moveTo(t[0][0], t[0][1]);
        for (let i = 1; i < t.length; i++) ctx.lineTo(t[i][0], t[i][1]);
        ctx.stroke();
      }
    }
    pista.classList.toggle("oculta", trazos.length > 0);
  }
  function pos(e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * N / r.width, (e.clientY - r.top) * N / r.height];
  }
  cv.addEventListener("pointerdown", (e) => {
    e.preventDefault(); cv.setPointerCapture(e.pointerId);
    actual = [pos(e)]; trazos.push(actual); dibujar();
  });
  cv.addEventListener("pointermove", (e) => {
    if (!actual) return;
    e.preventDefault(); actual.push(pos(e)); dibujar();
  });
  const soltar = () => { if (actual) { actual = null; enviar(); } };
  cv.addEventListener("pointerup", soltar);
  cv.addEventListener("pointercancel", soltar);

  document.getElementById("deshacer").onclick = () => { trazos.pop(); dibujar(); enviar(); };
  document.getElementById("borrar").onclick = () => { trazos = []; dibujar(); enviar(); };

  dibujar(); listo(); altura();
  new ResizeObserver(altura).observe(document.body);
  window.addEventListener("message", () => altura());
</script>
</body></html>
"""


@st.cache_resource
def preparar_lienzo():
    carpeta = Path(tempfile.gettempdir()) / "lienzo_digitos_componente"
    carpeta.mkdir(exist_ok=True)
    html = (
        LIENZO_HTML.replace("__PRIMARIO__", PRIMARIO)
        .replace("__OSCURO__", OSCURO)
        .replace("__CLARO__", CLARO)
        .replace("__SUAVE__", SUAVE)
        .replace("__GROSOR__", str(GROSOR_TRAZO))
    )
    archivo = carpeta / "index.html"
    if not archivo.exists() or archivo.read_text(encoding="utf-8") != html:
        archivo.write_text(html, encoding="utf-8")
    return components.declare_component("lienzo_digitos", path=str(carpeta))


lienzo = preparar_lienzo()


# =====================================================================
#  PREPROCESADO: deja la imagen "como las de MNIST"
# =====================================================================
def desplazar(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Mueve la imagen dy filas y dx columnas, rellenando con ceros."""
    alto, ancho = a.shape
    salida = np.zeros_like(a)
    ys_o = slice(max(0, -dy), alto - max(0, dy))
    ys_d = slice(max(0, dy), alto - max(0, -dy))
    xs_o = slice(max(0, -dx), ancho - max(0, dx))
    xs_d = slice(max(0, dx), ancho - max(0, -dx))
    salida[ys_d, xs_d] = a[ys_o, xs_o]
    return salida


def preparar_digito(gris: np.ndarray):
    """gris: matriz 2D uint8 con el trazo CLARO sobre fondo OSCURO.

    Hace lo mismo que se hizo para crear MNIST: recorta el dígito, lo escala a 20 px de lado
    mayor (sin deformarlo), lo pone en 28 × 28 y lo centra por su centro de masa.
    Devuelve (matriz 28×28 uint8, None) o (None, motivo).
    """
    a = gris.astype("float32")
    a[a < 0.12 * 255] = 0  # elimina el ruido de fondo
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return None, "vacia"
    alto, ancho = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if max(alto, ancho) < MIN_TRAZO_PX:
        return None, "pequena"

    recorte = Image.fromarray(a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1].astype("uint8"))
    escala = 20.0 / max(alto, ancho)
    nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    recorte = recorte.resize(nuevo, Image.Resampling.LANCZOS)

    r = np.asarray(recorte, dtype="float32")
    if r.max() > 0:
        r = r / r.max() * 255.0  # recupera el brillo máximo tras reducir

    lienzo28 = np.zeros((28, 28), dtype="float32")
    y0, x0 = (28 - r.shape[0]) // 2, (28 - r.shape[1]) // 2
    lienzo28[y0: y0 + r.shape[0], x0: x0 + r.shape[1]] = r

    masa = lienzo28.sum()
    cy = (np.arange(28)[:, None] * lienzo28).sum() / masa
    cx = (np.arange(28)[None, :] * lienzo28).sum() / masa
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    # El desplazamiento no debe sacar el trazo del cuadro (se recortaría el dígito)
    ys2, xs2 = np.nonzero(lienzo28)
    dy = int(np.clip(dy, -ys2.min(), 27 - ys2.max()))
    dx = int(np.clip(dx, -xs2.min(), 27 - xs2.max()))
    lienzo28 = desplazar(lienzo28, dy, dx)
    return np.clip(lienzo28, 0, 255).astype("uint8"), None


def gris_desde_lienzo(valor):
    """Convierte el valor que manda el lienzo (PNG en base64) en matriz de grises."""
    if not valor or not valor.get("img"):
        return None
    texto = valor["img"]
    if "," not in texto or len(texto) > 4_000_000:
        return None
    imagen = Image.open(io.BytesIO(base64.b64decode(texto.split(",", 1)[1])))
    return np.asarray(imagen.convert("L"))


def gris_desde_foto(datos: bytes) -> np.ndarray:
    """Foto o captura de un dígito -> matriz con trazo claro sobre fondo oscuro."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(datos)))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.split()[-1])
        img = base
    img.thumbnail((512, 512))
    g = ImageOps.autocontrast(img.convert("L"), cutoff=2)
    a = np.asarray(g, dtype="float32") / 255.0
    borde = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if borde.mean() > 0.5:  # fondo claro (papel): se invierte para dejar el trazo en blanco
        a = 1.0 - a
    a = np.clip((a - 0.35) / 0.65, 0, 1)
    return (a * 255).astype("uint8")


# =====================================================================
#  MODELO
# =====================================================================
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Devuelve (modelo, nombre_del_archivo). Si falla, lanza un error con la causa."""
    try:
        from tensorflow import keras
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"No se pudo importar TensorFlow: {e}") from e

    carpetas = [Path(__file__).resolve().parent, Path.cwd()]
    fallos = []
    for carpeta in carpetas:
        for nombre in NOMBRES_MODELO:
            ruta = carpeta / nombre
            if not ruta.is_file():
                continue
            try:
                modelo = keras.models.load_model(ruta, compile=False)
            except Exception as e:  # noqa: BLE001
                fallos.append(f"{nombre}: {e}")
                continue
            forma = tuple(modelo.input_shape[1:])
            if forma != (28, 28, 1):
                fallos.append(f"{nombre}: espera imágenes {forma}, no 28×28×1")
                continue
            return modelo, nombre
    if fallos:
        raise RuntimeError("El archivo del modelo no se pudo usar:\n" + "\n".join(fallos))
    raise FileNotFoundError("No se encontró " + " ni ".join(NOMBRES_MODELO))


@st.cache_data(show_spinner=False, max_entries=64)
def predecir(_modelo, pixeles: bytes) -> list:
    x = np.frombuffer(pixeles, dtype="uint8").astype("float32").reshape(1, 28, 28, 1) / 255.0
    return [float(p) for p in np.asarray(_modelo(x, training=False))[0]]


# =====================================================================
#  PRESENTACIÓN DEL RESULTADO
# =====================================================================
def png_base64(matriz: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(matriz).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def html_resultado(probs: list, pixeles: np.ndarray) -> str:
    orden = np.argsort(probs)[::-1]
    mejor, segundo = int(orden[0]), int(orden[1])
    conf = probs[mejor] * 100

    if conf >= CONFIANZA_ALTA:
        nota = "Muy seguro de este resultado."
    elif conf >= CONFIANZA_MEDIA:
        nota = f"Bastante seguro. La segunda opción sería el {segundo} ({probs[segundo] * 100:.0f} %)."
    else:
        nota = (
            f"No estoy seguro: podría ser {mejor} o {segundo}. "
            "Prueba a dibujarlo más grande y con trazo continuo."
        )

    barras = "".join(
        f'<div class="barra{" top" if i == mejor else ""}">'
        f'<div class="relleno" style="height:{max(probs[i] * 100, 3):.0f}%"></div></div>'
        for i in range(10)
    )
    etiquetas = "".join(f'<span{" class=top" if i == mejor else ""}>{i}</span>' for i in range(10))
    return f"""
    <div class="resultado">
        <p class="digito">{mejor}</p>
        <p class="confianza">{conf:.1f} % de confianza</p>
        <p class="nota">{nota}</p>
        <div class="barras">{barras}</div>
        <div class="etiquetas">{etiquetas}</div>
        <div class="vista">
            <img src="data:image/png;base64,{png_base64(pixeles)}" alt="Imagen de 28 por 28 píxeles">
            <small>Lo que ve el modelo<br>(28 × 28 píxeles)</small>
        </div>
    </div>
    """


MENSAJES_VACIO = {
    "vacia": "Aún no hay nada dibujado.<br>Tu resultado aparecerá aquí.",
    "pequena": "El trazo es muy pequeño.<br>Dibuja el dígito más grande.",
    "sin_trazo": "No encuentro un dígito en la imagen.<br>Prueba con un trazo más oscuro sobre fondo liso.",
}


# =====================================================================
#  PÁGINA
# =====================================================================
inyectar_css()

with st.sidebar:
    st.subheader("Cómo se usa")
    st.markdown(
        "1. Elige **Dibujar** o **Subir foto**.\n"
        "2. Dibuja un solo dígito, grande y centrado.\n"
        "3. El resultado aparece solo cuando sueltas el lápiz."
    )
    st.caption(
        "Consejo: un trazo continuo y bien visible funciona mejor que uno fino o entrecortado."
    )
    with st.expander("Detalles técnicos"):
        st.write(
            "Red neuronal convolucional entrenada con MNIST (70 000 dígitos escritos a mano): "
            "3 bloques de convolución (32, 64 y 128 filtros) con normalización, y capas densas "
            "de 256 y 128 neuronas. Entrada de 28 × 28 píxeles, salida de 10 probabilidades."
        )
        st.write(
            "Antes de predecir, el dibujo se recorta, se escala a 20 px y se centra por su "
            "centro de masa, igual que las imágenes con las que se entrenó."
        )
    st.caption("Tu dibujo no se guarda ni se envía a ningún servicio externo.")

st.markdown(
    """
    <div class="hero">
        <div class="etiqueta">Reconocimiento de caracteres</div>
        <h1>Detector de dígitos</h1>
        <p>Dibuja un número del 0 al 9 y una red neuronal intentará adivinar cuál es.</p>
        <div class="linea"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    modelo, archivo_modelo = cargar_modelo()
except Exception as error:  # noqa: BLE001
    st.error("No pude cargar el modelo, así que la app no puede predecir todavía.")
    st.markdown(
        "**Qué revisar**\n"
        f"1. Que `{NOMBRES_MODELO[0]}` o `{NOMBRES_MODELO[1]}` esté en el repositorio, "
        "junto a este archivo.\n"
        "2. Que `requirements.txt` incluya `tensorflow`.\n"
        "3. Que la app use Python 3.12 (en Streamlit Cloud: *Settings → Advanced settings*)."
    )
    with st.expander("Detalle del error"):
        st.code(str(error))
    st.stop()

with st.container(key="tarjeta_digito"):
    fuente = st.radio(
        "¿Cómo quieres darme el dígito?",
        ["Dibujar", "Subir foto"],
        horizontal=True,
    )
    izquierda, derecha = st.columns(2, gap="large")

    gris, motivo = None, "vacia"
    with izquierda:
        if fuente == "Dibujar":
            valor = lienzo(key="lienzo", default=None)
            gris = gris_desde_lienzo(valor)
        else:
            archivo = st.file_uploader(
                "Foto o captura de un solo dígito",
                type=["png", "jpg", "jpeg", "webp"],
                help="Funciona mejor con un trazo grueso sobre fondo liso, por ejemplo marcador sobre papel.",
            )
            if archivo is not None:
                try:
                    gris = gris_desde_foto(archivo.getvalue())
                    st.image(archivo, caption="Tu imagen", use_container_width=True)
                except Exception:  # noqa: BLE001
                    st.warning("No pude abrir esa imagen. Prueba con un PNG o JPG.")
                    motivo = "sin_trazo"

    with derecha:
        pixeles = None
        if gris is not None:
            pixeles, motivo = preparar_digito(gris)
            if pixeles is None and fuente == "Subir foto":
                motivo = "sin_trazo" if motivo == "vacia" else motivo
        if pixeles is None:
            st.markdown(f'<div class="vacio">{MENSAJES_VACIO[motivo]}</div>', unsafe_allow_html=True)
        else:
            probs = predecir(modelo, pixeles.tobytes())
            st.markdown(html_resultado(probs, pixeles), unsafe_allow_html=True)

st.caption(f"Modelo: {archivo_modelo} · reconoce un dígito a la vez.")
