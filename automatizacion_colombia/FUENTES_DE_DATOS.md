# Fuentes de Datos: Costos Laborales y Automatización en Colombia

## Resumen de Prioridades

| Prioridad | Dataset | Para qué | Descarga |
|-----------|---------|----------|----------|
| **CRITICA** | GEIH (DANE) | Salarios, informalidad, ocupaciones por sector | Registro gratuito |
| **CRITICA** | EAM (DANE) | Costos laborales firma, inversión en maquinaria | Registro gratuito |
| **CRITICA** | Penn World Table 11.0 | TFP, stock de capital, comparaciones internacionales | Descarga directa |
| **CRITICA** | ILOSTAT | Productividad laboral por hora por país | Descarga directa |
| **ALTA** | EDIT X (DANE) | Innovación tecnológica en manufactura | Registro gratuito |
| **ALTA** | World Bank WDI | I+D como % PIB, PIB por trabajador | Descarga directa |
| **ALTA** | OECD Taxing Wages | Cuña fiscal, costos laborales comparados | Descarga directa |
| **ALTA** | Cuentas Nacionales (DANE) | Productividad laboral sectorial, VA, acervos de capital | Descarga directa |
| **MEDIA** | Frey & Osborne (2017) | Probabilidades de automatización por ocupación | PDF (Apéndice A) |
| **MEDIA** | IFR / Our World in Data | Densidad de robots por país | CSV gratuito (OWID) |
| **MEDIA** | ILIA 2024 | Índice de preparación IA en LatAm | PDF |
| **MEDIA** | BID/IDB Automation | Riesgo de automatización en LatAm | PDF + datos abiertos |

---

## 1. FUENTES DANE (Requieren registro gratuito)

### Registro
- **URL:** https://microdatos.dane.gov.co/auth/register
- Crear cuenta gratuita. Se requiere validación reCAPTCHA para descargas.

---

### 1.1 GEIH — Gran Encuesta Integrada de Hogares

**Propósito:** Análisis a nivel de trabajador: salarios, horas, formalidad, ocupación, sector económico. Base para construir medidas de costo laboral por sector y mapear probabilidades de automatización por ocupación.

**URLs de descarga (microdatos):**
- 2023: https://microdatos.dane.gov.co/index.php/catalog/782/get-microdata
- 2024: https://microdatos.dane.gov.co/index.php/catalog/819/get-microdata
- 2025: https://microdatos.dane.gov.co/index.php/catalog/853/get-microdata

**Formato:** ZIP mensuales (Enero.zip a Diciembre.zip). Dentro: CSV y DTA (Stata).

**Variables clave:**
- Clasificación ocupacional (mapeable a Frey & Osborne vía ISCO)
- Códigos CIIU (sector económico)
- Ingreso laboral (empleo principal y secundario)
- Horas trabajadas
- Tipo de empleo (formal/informal, asalariado/independiente)
- Afiliación a seguridad social (salud, pensión)
- Nivel educativo
- Tamaño de empresa
- Rama de actividad económica

**Años disponibles:** 2001-2025 (continua).

---

### 1.2 EAM — Encuesta Anual Manufacturera

**Propósito:** Análisis a nivel de firma manufacturera: costos laborales totales (salarios + prestaciones), inversión en maquinaria y equipo (proxy de automatización), valor agregado, productividad.

**URLs de descarga (microdatos):**
- 2023: https://microdatos.dane.gov.co/index.php/catalog/871/get-microdata
- 2022: https://microdatos.dane.gov.co/index.php/catalog/836/get-microdata
- 2021: https://microdatos.dane.gov.co/index.php/catalog/802/get-microdata

**Página principal DANE:**
https://www.dane.gov.co/index.php/estadisticas-por-tema/industria/encuesta-anual-manufacturera-enam

**Históricos:**
https://www.dane.gov.co/index.php/estadisticas-por-tema/industria/encuesta-anual-manufacturera-enam/eam-historicos

**Formato:** ZIP con archivos XLSX. Diccionario de datos en Excel.

**Variables clave (6 módulos):**
- **Módulo II — Personal y nómina:** Número de empleados (permanentes, temporales, tercerizados), salarios totales, contribuciones a seguridad social, otros costos laborales
- **Módulo III — Costos industriales y activos fijos:** Inversión en maquinaria y equipo (PROXY PRINCIPAL DE AUTOMATIZACIÓN), depreciación, formación bruta de capital fijo
- **Módulo V — Productos y materias primas:** Valor de producción, consumo intermedio
- **Módulo VI — Ingresos no industriales y TIC:** Indicadores de adopción de TIC
- **Derivado:** Valor agregado = producción bruta − consumo intermedio

**Años disponibles:** 1992-2023 (anónimizados).

**NOTA:** La EAM y la EDIT comparten el marco muestral del censo manufacturero del DANE, lo que permite vincular firmas entre ambas encuestas.

---

### 1.3 EDIT X — Encuesta de Desarrollo e Innovación Tecnológica

**Propósito:** Inversión en tecnología e innovación a nivel de firma. La fuente más directa para medir adopción de automatización: gasto en maquinaria para innovación, software, I+D.

**URLs de descarga (microdatos):**
- EDIT X (Manufactura, 2019-2020): https://microdatos.dane.gov.co/index.php/catalog/868/get-microdata
- EDITS VIII (Servicios y comercio, 2020-2021): https://microdatos.dane.gov.co/index.php/catalog/867/get-microdata

**Página principal DANE:**
https://www.dane.gov.co/index.php/estadisticas-por-tema/tecnologia-e-innovacion/encuesta-de-desarrollo-e-innovacion-tecnologica-edit

**Formato:** ZIP con XLSX por capítulo.

**Variables clave:**
- **Cap. II — Inversión en ACTI:** Gasto en I+D interna, adquisición de maquinaria y equipo para innovación, transferencia tecnológica, adquisición de software, ingeniería y diseño industrial
- **Cap. III — Financiamiento:** Fuentes de financiamiento para innovación
- **Cap. IV — Personal en ACTI:** Número y nivel educativo de trabajadores dedicados a innovación
- **Cap. V — Resultados de innovación:** Tipos de innovaciones (producto, proceso, organizacional)
- **Cap. VII — Gestión empresarial:** Obstáculos a la innovación, certificaciones de calidad

**Años disponibles:** EDIT I a EDIT X para manufactura (2000s-2020). EDITS I a VIII para servicios. Cada edición cubre un período bienal. EDIT XI (2021-2022) NO está publicada aún.

---

### 1.4 Cuentas Nacionales — Productividad y PIB Sectorial

**Propósito:** Productividad laboral por sector, valor agregado sectorial, stock de capital, PTF. Contexto macroeconómico.

**URLs de descarga directa (Excel, NO requieren registro):**

**Productividad:**
- Productividad laboral: https://www.dane.gov.co/files/operaciones/PTF/anex-PTF-ProductividadLaboral-2024.xlsx
- Productividad Total de los Factores: https://www.dane.gov.co/files/operaciones/PTF/anex-PTF-Productividad-2024.xlsx
- Acervos de capital: https://www.dane.gov.co/files/operaciones/PTF/anex-PTF-AcervosCapital-2023.xlsx
- Página principal: https://www.dane.gov.co/index.php/estadisticas-por-tema/cuentas-nacionales/productividad

**PIB por sector (Cuentas Nacionales Anuales):**
- Agregados macroeconómicos 2005-2024p: https://www.dane.gov.co/files/operaciones/PIB/anex-CuentasNalANuales-AgreMacroeconomicos-2024p.xlsx
- Oferta-utilización precios constantes: https://www.dane.gov.co/files/operaciones/PIB/anex-CuentasNalANuales-OfertaUtilizacionPreciosConstantes-2024p.xlsx
- Oferta-utilización precios corrientes: https://www.dane.gov.co/files/operaciones/PIB/anex-CuentasNalANuales-OfertaUtilizacionPreciosCorrientes-2024p.xlsx
- Cuentas económicas integradas: https://www.dane.gov.co/files/operaciones/PIB/anex-CuentasNalAnuales-CuentasEconomicasIntegradas-2024p.xlsx
- Página principal: https://www.dane.gov.co/index.php/estadisticas-por-tema/cuentas-nacionales/cuentas-nacionales-anuales

**PIB trimestral por sector:**
- Precios constantes Q4 2025: https://www.dane.gov.co/files/operaciones/PIB/anex-ProduccionConstantes-IVtrim2025.xlsx
- Precios corrientes Q4 2025: https://www.dane.gov.co/files/operaciones/PIB/anex-ProduccionCorriente-IVtrim2025.xlsx
- Página técnica: https://www.dane.gov.co/index.php/estadisticas-por-tema/cuentas-nacionales/cuentas-nacionales-trimestrales/pib-informacion-tecnica

**Formato:** Excel (.xlsx). Datos agregados sectoriales.

**Variables clave:**
- Valor agregado por 12 agrupaciones de actividades económicas (CIIU Rev. 4)
- Productividad laboral = VA / empleo (o por hora trabajada)
- Descomposición de PTF
- Acervos de capital por sector
- Participación de la compensación laboral en el VA (enfoque ingreso)

---

## 2. FUENTES INTERNACIONALES (Descarga directa, gratuitas)

---

### 2.1 Penn World Table 11.0

**Propósito:** Comparaciones internacionales de TFP, stock de capital, productividad. Datos para 185 países, 1950-2023.

**URLs de descarga directa:**
- Excel: https://dataverse.nl/api/access/datafile/554105
- Stata: https://dataverse.nl/api/access/datafile/554030
- Página principal: https://www.rug.nl/ggdc/productivity/pwt/

**Formato:** Excel (.xlsx), Stata (.dta). Licencia CC BY 4.0.

**Variables clave:**
- `ctfp` — Nivel de PTF a PPP corrientes (USA=1)
- `rtfpna` — PTF a precios nacionales constantes (índice, 2017=1)
- `rnna` / `rkna` — Stock de capital a precios nacionales constantes
- `cn` — Stock de capital a PPP corrientes
- Descomposiciones de contabilidad del crecimiento

---

### 2.2 ILOSTAT — Productividad Laboral por País

**Propósito:** PIB por hora trabajada y por persona empleada. Comparaciones internacionales de productividad.

**URLs de descarga:**
- Descarga masiva (CSV): https://ilostat.ilo.org/data/bulk/
- Excel directo — PIB por hora: https://ilo.org/ilostat-files/Documents/Excel/Indicator/LAP_2GDP_NOC_RT_A_EN.xlsx
- Portal de datos: https://data.ilo.org/
- Tema productividad: https://ilostat.ilo.org/topics/labour-productivity/

**Formato:** CSV (descarga masiva), Excel (.xlsx). También API SDMX y paquete R (`Rilostat`).

**Acceso:** Completamente abierto, sin registro.

**Variables clave:**
- `LAP_2GDP` — PIB por hora trabajada
- PIB por persona empleada
- Indicador ODS 8.2.1 (tasa de crecimiento del PIB real por persona empleada)

---

### 2.3 OECD Taxing Wages — Cuña Fiscal

**Propósito:** Cuña fiscal sobre el trabajo (tax wedge), descomposición de costos laborales para Colombia y países comparadores.

**URLs:**
- Data Explorer: https://data-explorer.oecd.org/vis?df%5Bds%5D=DisseminateFinalDMZ&df%5Bid%5D=DSD_TAX_WAGES_DECOMP%40DF_TW_DECOMP&df%5Bag%5D=OECD.CTP.TPS
- Indicador: https://www.oecd.org/en/data/indicators/tax-wedge.html
- Nota Colombia (PDF): https://www.oecd.org/content/dam/oecd/en/publications/reports/2025/04/taxing-wages-2025-country-notes_16d47563/colombia_511db0ad/4d618d8e-en.pdf

**Formato:** CSV vía Data Explorer, API SDMX.

**NOTA IMPORTANTE:** La cuña fiscal de Colombia aparece como 0% en OECD porque las contribuciones a seguridad social se clasifican como "pagos obligatorios no tributarios" (NTCPs). Hay que sumar manualmente los aportes parafiscales y de seguridad social.

---

### 2.4 World Bank WDI

**Propósito:** I+D como % del PIB, PIB por trabajador, fuerza laboral total. Comparaciones internacionales.

**URLs de descarga directa (cada indicador tiene CSV/Excel):**
- PIB por persona empleada (PPP constante 2021): https://data.worldbank.org/indicator/SL.GDP.PCAP.EM.KD
- Gasto en I+D (% del PIB): https://data.worldbank.org/indicator/GB.XPD.RSDV.GD.ZS
- Fuerza laboral total: https://data.worldbank.org/indicator/SL.TLF.TOTL.IN
- Descarga masiva WDI: https://datacatalog.worldbank.org/search/dataset/0037712/world-development-indicators
- DataBank interface: https://databank.worldbank.org/source/world-development-indicators

**Formato:** CSV, Excel, API. Datos abiertos, sin registro.

---

### 2.5 IFR / Our World in Data — Densidad de Robots

**Propósito:** Robots industriales instalados por país, densidad de robots por 10,000 trabajadores manufactureros.

**NOTA:** Los datos completos del IFR cuestan EUR 1,850. Usar la alternativa gratuita:

**Alternativa gratuita — Our World in Data (fuente: IFR vía Stanford AI Index):**
- Gráfico interactivo: https://ourworldindata.org/grapher/annual-industrial-robots-installed
- GitHub datasets: https://github.com/owid/owid-datasets
- Formato: CSV, licencia CC BY.

**Resúmenes ejecutivos IFR (gratuitos):**
- 2025: https://ifr.org/img/worldrobotics/Executive_Summary_WR_2025_Industrial_Robots.pdf
- 2024: https://ifr.org/img/worldrobotics/Executive_Summary_WR_2024_Industrial_Robots.pdf

**Stanford AI Index 2025 (datos en Kaggle, incluye datos IFR):**
- https://www.kaggle.com/datasets/paultimothymooney/ai-index-report-2025

---

## 3. PAPERS Y ESTUDIOS ESPECÍFICOS

---

### 3.1 Fedesarrollo / Mejía y Pabón — Automatización en países andinos

**PDF directo (IDB):**
https://publications.iadb.org/publications/spanish/document/COVID-19-y-riesgo-de-automatizacion-en-el-mercado-laboral-de-los-paises-andinos.pdf

**Hallazgo clave:** 58% de trabajadores colombianos en alto riesgo de automatización. Sectores más expuestos: servicios (18%), comercio (17%), agricultura (16%), tareas administrativas (8%).

---

### 3.2 Morales, Atis y Fajardo (2023) — Riesgo de automatización en Colombia

**PDF directo (Dialnet):**
https://dialnet.unirioja.es/descarga/articulo/10231079.pdf

**HTML completo (Redalyc):**
https://www.redalyc.org/journal/909/90978510010/html/

**SciELO Colombia:**
http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S0121-68052023000200159

**Hallazgo clave:** 37% de la fuerza laboral colombiana en alto riesgo, 55% en riesgo medio, 7% en riesgo bajo. Usa GEIH 2019 + probabilidades Frey & Osborne mapeadas a ISCO-1968. Tabla A1 en apéndice con asignaciones de probabilidad por ocupación.

---

### 3.3 OECD/GPAI (2025) — "Voces del cambio" (IA Generativa en LatAm)

**PDF directo:**
https://wp.oecd.ai/app/uploads/2025/05/Voces-del-cambio-la-IA-generativa-y-la-transformacion-del-trabajo-en-America-Latina.pdf

**Hallazgo clave:** 2-5% del empleo en LatAm enfrenta riesgo de automatización por IA generativa; 8-12% podría beneficiarse de transformación productiva.

---

### 3.4 Acemoglu y Restrepo (2020) — "Robots and Jobs" (JPE)

**PDF (MIT):**
https://shapingwork.mit.edu/wp-content/uploads/2023/10/Robots-and-Jobs-Evidence-from-US-Labor-Markets.p.pdf

**Datos de replicación (Stata do-files + data):**
https://economics.mit.edu/people/faculty/daron-acemoglu/data-archive

**Hallazgo clave:** Un robot adicional por mil trabajadores reduce empleo en 0.2 pp y salarios en 0.42%.

---

### 3.5 Cheng et al. (2021) — Elasticidad de sustitución (σ ≈ 3.8)

**PDF (Philadelphia Fed):**
https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2021/wp21-11.pdf

**Hallazgo clave:** Elasticidad de sustitución entre capital de automatización y trabajo = 3.8. Usa 1,618 firmas manufactureras chinas bajo "Made in China 2025".

---

### 3.6 Frey & Osborne (2017) — Probabilidades de automatización por ocupación

**PDF (Oxford Martin School):**
https://oms-www.files.svdcdn.com/production/downloads/academic/The_Future_of_Employment.pdf

**Datos:** El Apéndice A del PDF contiene las probabilidades de automatización para las 702 ocupaciones SOC de EE.UU. NO hay CSV oficial. Buscar en GitHub "frey osborne automation" para versiones extraídas por la comunidad.

**Adaptación UK (ONS) con datos estructurados:**
https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/articles/theprobabilityofautomationinengland/2011and2017

---

### 3.7 CONPES 4144 — Política Nacional de IA

**PDF completo:**
https://colaboracion.dnp.gov.co/CDT/Conpes/Econ%C3%B3micos/4144.pdf

**Anexo A (Plan de Acción, Excel):**
Disponible en: https://sisconpes.dnp.gov.co (buscar CONPES 4144)

---

### 3.8 Asobancaria (2024) — Adopción de IA en sector financiero

**Informe de Gestión Gremial 2024:**
https://www.asobancaria.com/wp-content/uploads/2025/08/IGG-2024-20.pdf

**Informe de Tipificación 2024:**
https://www.asobancaria.com/wp-content/uploads/2025/09/INFORME-TIPIFICACION-2024.pdf

**Hallazgo clave:** 73% de entidades financieras han implementado IA; productividad +41.8%.

---

### 3.9 ANIF — Costos laborales no salariales

**Estudio ANIF/ACOPI sobre sobrecostos (post Ley 1607):**
https://acopi.org.co/wp-content/uploads/2018/05/Anif-ACOPI-Sobrecostos.pdf

**Análisis ANIF 2025 sobre reforma laboral:**
https://consultorsalud.com/wp-content/uploads/2025/06/la-reforma-laboral-que-pretende-mejorar-la-condicion-de-los-trabajadores-formales-pero-que-hay-del-60-restante-1-1.pdf

**Biblioteca ANIF general:** https://www.anif.co/Biblioteca

**Hallazgo clave:** Costos no salariales = 53% de la nómina. Un incremento de 1% en costos no salariales → +0.1% desempleo, −0.4% formalidad.

---

### 3.10 ILIA 2024 — Índice Latinoamericano de IA

**PDF del reporte:**
https://indicelatam.cl/wp-content/uploads/2024/09/ILIA_2024.pdf

**ILIA 2025 (edición más reciente):**
https://indicelatam.cl/home-2025-en/

---

### 3.11 BID — Automatización en América Latina

**Paper principal (Brambilla, Cesar, Falcone & Gasparini):**
https://publications.iadb.org/en/automation-trends-and-labor-markets-latin-america

**Portal de datos abiertos IDB:**
- https://data.iadb.org/
- Indicadores sociales LatAm: https://data.iadb.org/dataset/social-indicators-of-latin-america-and-the-caribbean

---

### 3.12 WEF Future of Jobs Report 2025

**PDF completo:**
https://reports.weforum.org/docs/WEF_Future_of_Jobs_Report_2025.pdf

**Hallazgo clave:** +170M empleos creados, −92M destruidos → saldo neto +78M para 2030.

---

## 4. ORDEN DE DESCARGA RECOMENDADO

### Paso 1 — Descargas directas (sin registro, hacer inmediatamente):
1. Penn World Table 11.0 (Excel)
2. ILOSTAT productividad por hora (Excel)
3. World Bank WDI — I+D % PIB + PIB por trabajador (CSV)
4. Cuentas Nacionales DANE — Productividad laboral + PIB sectorial + Acervos de capital (Excel)
5. OECD Taxing Wages — Colombia country note (PDF) + data explorer (CSV)
6. Our World in Data — Robots instalados por país (CSV)
7. Stanford AI Index 2025 en Kaggle (ZIP)

### Paso 2 — Registrarse en microdatos.dane.gov.co y descargar:
8. GEIH 2023 + 2024 (microdatos mensuales)
9. EAM 2021, 2022, 2023 (microdatos de establecimientos)
10. EDIT X manufactura 2019-2020 + EDITS VIII servicios 2020-2021

### Paso 3 — Descargar papers y reportes (PDFs):
11. Todos los PDFs listados en la sección 3

---

## 5. EJERCICIOS ECONOMÉTRICOS PLANEADOS

### Con GEIH:
- Construcción de índice de riesgo de automatización por ocupación (mapeo Frey & Osborne → ISCO → clasificación colombiana)
- Análisis de composición sectorial del empleo formal vs informal
- Cálculo de costos laborales implícitos por sector (salario + proxy de prestaciones por formalidad)
- Regresión: P(automatización alta) = f(sector, formalidad, educación, ingreso, tamaño firma)

### Con EAM:
- Ratio costos laborales / valor agregado por subsector manufacturero
- Inversión en maquinaria y equipo como % del VA (proxy de automatización)
- Panel econométrico: ΔInversión_maquinaria = f(ΔCostos_laborales, productividad, tamaño, sector)
- Test de causalidad de Granger: costos laborales → inversión en capital

### Con EDIT + EAM (merge):
- Determinantes de la innovación de proceso: P(innova_proceso) = f(costos_laborales, tamaño, sector, exporta)
- Gasto en automatización como proporción del gasto total en innovación

### Con datos internacionales:
- Panel de países: Densidad_robots = f(costos_laborales, productividad, I+D/PIB, PIB_pc)
- Dónde se ubica Colombia en la curva de adopción
- Simulación: proyección de densidad de robots en Colombia bajo diferentes escenarios de costos laborales

### Análisis sectorial integrado:
- Matriz de vulnerabilidad: (potencial técnico de automatización) × (incentivo económico por costos laborales) × (formalidad) → índice de riesgo compuesto por sector
