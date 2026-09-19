"""
legal.py — Terminos de Uso y Politica de Privacidad (RGPD)
WealthView · wealthview.es
"""
import streamlit as st
from modules.styles import (
    SURFACE, SURFACE_2, BORDER, GOLD, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED
)


def render_legal():
    st.markdown(f"""
    <div style='max-width:760px;margin:0 auto;padding:20px 0;'>
        <div style='font-size:11px;color:{GOLD};text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:10px;'>WealthView</div>
        <h1 style='font-size:28px;font-weight:700;color:{TEXT_PRIMARY};margin:0 0 6px 0;'>
            Aviso Legal</h1>
        <p style='color:{TEXT_MUTED};font-size:12px;margin:0 0 32px 0;'>
            Ultima actualizacion: julio 2025 · Aplicable en la Union Europea
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_terms, tab_privacy = st.tabs(["Condiciones de Uso", "Politica de Privacidad"])

    # ── TAB 1: Condiciones de Uso ──────────────────────────────────────────────
    with tab_terms:
        st.markdown(f"""
<div style='max-width:760px;margin:0 auto;color:{TEXT_SECONDARY};font-size:14px;line-height:1.85;'>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>1. Objeto y aceptacion</h3>
<p>Las presentes Condiciones de Uso regulan el acceso y la utilizacion del servicio WealthView,
disponible en <b>wealthview.es</b> (en adelante, "el Servicio"). Al registrarte o utilizar
el Servicio, aceptas quedar vinculado por estas condiciones. Si no las aceptas, no debes
utilizar el Servicio.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>2. Descripcion del servicio</h3>
<p>WealthView es una plataforma de analisis y gestion de carteras de inversion que ofrece
herramientas de seguimiento de activos, analisis cuantitativo, alertas de precio y generacion
de informes. El Servicio tiene caracter <b>informativo y analitico</b>; no constituye en ningun
caso asesoramiento financiero, de inversion, legal o fiscal.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>3. Advertencia de inversion</h3>
<p style='background:rgba(201,168,76,0.08);border-left:3px solid {GOLD};padding:12px 16px;
          border-radius:0 6px 6px 0;'>
<b>AVISO IMPORTANTE:</b> La informacion, analisis y calculos proporcionados por WealthView
tienen un proposito exclusivamente educativo e informativo. No constituyen recomendaciones de
compra o venta de valores ni asesoramiento financiero regulado. Las rentabilidades pasadas
no garantizan resultados futuros. Toda decision de inversion es responsabilidad exclusiva
del usuario. Consulta con un asesor financiero habilitado antes de tomar decisiones de inversion.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>4. Registro y cuenta de usuario</h3>
<p>Para acceder al Servicio es necesario crear una cuenta con un nombre de usuario y contrasena.
El usuario es responsable de mantener la confidencialidad de sus credenciales y de todas las
actividades realizadas desde su cuenta. Debes notificarnos inmediatamente cualquier uso no
autorizado de tu cuenta.</p>
<p>El registro esta permitido unicamente para personas mayores de 18 anos. Al registrarte,
declaras que la informacion proporcionada es veraz y actualizada.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>5. Uso aceptable</h3>
<p>El usuario se compromete a utilizar el Servicio de conformidad con la ley, la moral y
el orden publico. Queda expresamente prohibido:</p>
<ul style='padding-left:20px;'>
<li>Intentar acceder a cuentas de otros usuarios.</li>
<li>Utilizar el Servicio para actividades ilegales o fraudulentas.</li>
<li>Realizar ingenieria inversa, descompilar o intentar extraer el codigo fuente.</li>
<li>Sobrecargar o interferir con la infraestructura tecnica del Servicio.</li>
<li>Revender o sublicenciar el acceso al Servicio a terceros.</li>
</ul>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>6. Disponibilidad del servicio</h3>
<p>El Servicio se presta "tal cual" y "segun disponibilidad". No garantizamos disponibilidad
ininterrumpida ni libre de errores. Podemos suspender o modificar el Servicio en cualquier
momento, con o sin previo aviso, por razones tecnicas, de mantenimiento o de negocio.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>7. Limitacion de responsabilidad</h3>
<p>En la maxima medida permitida por la ley aplicable, WealthView no sera responsable de
danos directos, indirectos, incidentales, especiales o consecuentes derivados del uso o
la imposibilidad de uso del Servicio, incluyendo perdidas economicas derivadas de decisiones
de inversion basadas en la informacion proporcionada.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>8. Propiedad intelectual</h3>
<p>Todos los derechos de propiedad intelectual sobre el Servicio, incluyendo su diseno,
codigo, logotipos y contenidos, pertenecen a WealthView o a sus licenciantes. El usuario
recibe una licencia limitada, no exclusiva e intransferible para usar el Servicio segun
estas condiciones.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>9. Modificacion de las condiciones</h3>
<p>Nos reservamos el derecho a modificar estas condiciones en cualquier momento. Los cambios
sustanciales seran comunicados mediante aviso en la plataforma. El uso continuado del Servicio
tras la notificacion implica la aceptacion de las nuevas condiciones.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>10. Ley aplicable y jurisdiccion</h3>
<p>Estas condiciones se rigen por la ley espanola. Para cualquier controversia derivada de
su interpretacion o aplicacion, las partes se someten a los juzgados y tribunales de Espana,
con renuncia expresa a cualquier otro fuero.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>11. Contacto</h3>
<p>Para cualquier consulta sobre estas condiciones puedes contactarnos en:
<b style='color:{TEXT_PRIMARY};'>hola@wealthview.es</b></p>

</div>
""", unsafe_allow_html=True)

    # ── TAB 2: Politica de Privacidad ─────────────────────────────────────────
    with tab_privacy:
        st.markdown(f"""
<div style='max-width:760px;margin:0 auto;color:{TEXT_SECONDARY};font-size:14px;line-height:1.85;'>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>1. Responsable del tratamiento</h3>
<p>El responsable del tratamiento de los datos personales recogidos a traves de WealthView es:</p>
<p style='background:{SURFACE_2};border:1px solid {BORDER};border-radius:8px;
          padding:14px 18px;font-size:13px;'>
<b style='color:{TEXT_PRIMARY};'>WealthView</b><br>
wealthview.es<br>
Contacto: <b>hola@wealthview.es</b>
</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>2. Datos que recogemos</h3>
<p>Recogemos unicamente los datos estrictamente necesarios para prestar el Servicio:</p>
<ul style='padding-left:20px;'>
<li><b style='color:{TEXT_PRIMARY};'>Datos de cuenta:</b> nombre de usuario y contrasena
(almacenada de forma cifrada con bcrypt; nunca en texto plano).</li>
<li><b style='color:{TEXT_PRIMARY};'>Datos de portfolio:</b> tickers de activos, numero de
acciones, precios de compra y cualquier otra informacion que el usuario introduzca
voluntariamente.</li>
<li><b style='color:{TEXT_PRIMARY};'>Datos tecnicos:</b> registros de acceso (fecha, hora,
IP de origen) para la seguridad y prevencion de fraude.</li>
</ul>
<p>No recogemos datos de pago, datos biometricos ni datos especialmente protegidos.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>3. Finalidad y base legal</h3>
<p>Tratamos tus datos para las siguientes finalidades:</p>
<ul style='padding-left:20px;'>
<li><b style='color:{TEXT_PRIMARY};'>Prestacion del Servicio</b> (Art. 6.1.b RGPD — ejecucion
de contrato): gestion de cuenta, almacenamiento de portfolio y prestacion de las funcionalidades
de la plataforma.</li>
<li><b style='color:{TEXT_PRIMARY};'>Seguridad</b> (Art. 6.1.f RGPD — interes legitimo):
prevencion de accesos no autorizados, deteccion de fraude y mantenimiento de registros de
acceso.</li>
</ul>
<p>No utilizamos tus datos para publicidad ni los cedemos a terceros con fines comerciales.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>4. Conservacion de los datos</h3>
<p>Conservamos tus datos mientras tu cuenta este activa. Si solicitas la eliminacion de tu cuenta,
suprimiremos tus datos personales en un plazo maximo de 30 dias, salvo que la ley nos obligue
a conservarlos durante un periodo determinado.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>5. Terceros y transferencias internacionales</h3>
<p>El Servicio utiliza las siguientes fuentes de datos de mercado externas, a las que se
transmiten unicamente los tickers que el usuario consulta (no datos personales):</p>
<ul style='padding-left:20px;'>
<li>Yahoo Finance (Sunnyvale, CA, USA)</li>
<li>Polygon.io (Nueva York, USA)</li>
<li>Financial Modeling Prep (USA)</li>
<li>FRED — Federal Reserve Bank of St. Louis (USA)</li>
</ul>
<p>Estas transferencias se realizan al amparo de las clausulas contractuales tipo de la
Comision Europea o de la excepcion por necesidad contractual (Art. 49.1.b RGPD).</p>
<p>El servidor del Servicio esta ubicado en la Union Europea.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>6. Tus derechos</h3>
<p>Bajo el RGPD, tienes derecho a:</p>
<ul style='padding-left:20px;'>
<li><b style='color:{TEXT_PRIMARY};'>Acceso:</b> solicitar una copia de los datos que tratamos sobre ti.</li>
<li><b style='color:{TEXT_PRIMARY};'>Rectificacion:</b> corregir datos inexactos o incompletos.</li>
<li><b style='color:{TEXT_PRIMARY};'>Supresion:</b> solicitar la eliminacion de tus datos ("derecho al olvido").</li>
<li><b style='color:{TEXT_PRIMARY};'>Portabilidad:</b> recibir tus datos en un formato estructurado y legible.</li>
<li><b style='color:{TEXT_PRIMARY};'>Oposicion y limitacion:</b> oponerte a determinados tratamientos o
solicitar su limitacion.</li>
</ul>
<p>Para ejercer cualquiera de estos derechos, contacta en <b style='color:{TEXT_PRIMARY};'>hola@wealthview.es</b>.
Responderemos en un plazo maximo de 30 dias.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>7. Reclamaciones</h3>
<p>Si consideras que el tratamiento de tus datos no es conforme a la normativa, tienes derecho
a presentar una reclamacion ante la <b style='color:{TEXT_PRIMARY};'>Agencia Espanola de
Proteccion de Datos (AEPD)</b> — <a href='https://www.aepd.es' target='_blank'
style='color:{GOLD};'>www.aepd.es</a>.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>8. Seguridad</h3>
<p>Aplicamos medidas tecnicas y organizativas adecuadas para proteger tus datos, incluyendo
cifrado de contrasenas (bcrypt), comunicaciones HTTPS, control de acceso por sesion,
rate limiting y registros de auditoria.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>9. Cookies</h3>
<p>WealthView no utiliza cookies de seguimiento ni publicidad. La sesion de usuario se
gestiona mediante tokens de sesion tecnicamente necesarios para el funcionamiento del Servicio.</p>

<h3 style='color:{TEXT_PRIMARY};font-size:16px;margin-top:24px;'>10. Modificaciones</h3>
<p>Podemos actualizar esta politica para reflejar cambios en nuestras practicas o en la
normativa aplicable. Notificaremos los cambios significativos mediante aviso en la plataforma.</p>

</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center;color:{TEXT_MUTED};font-size:11px;'>"
        f"WealthView · wealthview.es · hola@wealthview.es</p>",
        unsafe_allow_html=True
    )
