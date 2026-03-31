# Estructura de la Investigación

## Título
**Dinámica de Sustitución Tecnológica en Colombia: El Impacto de los Costos Laborales en la Aceleración de la Automatización y la Inteligencia Artificial**

## Autores
- Cristian Espinal Maya (cespinalm@eafit.edu.co, ORCID: 0009-0000-1009-8388) — Universidad EAFIT
- Santiago Jiménez Londoño (sjimenez@eafit.edu.co, ORCID: 0009-0007-9862-7133) — Universidad EAFIT

## Tesis Central
Los altos costos laborales no salariales en Colombia (46-58% sobre el salario base) crean un incentivo estructural para la automatización que, combinado con los avances en IA generativa, acelerará la sustitución capital-trabajo de forma diferenciada por sectores, siendo más brusca en sectores de alta formalidad, altos salarios relativos y alta proporción de tareas rutinarias.

---

## Estructura del Paper

### 1. Introducción y Planteamiento
- Contexto global de la automatización e IA
- Particularidad de economías emergentes vs. desarrolladas
- Tesis: automatización como mecanismo defensivo del sector corporativo frente a costos laborales
- Contribución: primer análisis integrado que vincula costos laborales colombianos con riesgo de automatización sectorial
- Preview de resultados

### 2. Marco Teórico
#### 2.1 Modelo de Tareas de Acemoglu y Restrepo (2019, 2020)
- Efecto desplazamiento, efecto reinstauración, efecto productividad
- Elasticidad de sustitución σ > 1 como condición para sustitución acelerada
- Evidencia empírica: σ ≈ 3.8 (Cheng et al. 2021), 2.14-3.27 (Adachi)

#### 2.2 Tecnologías "Regulares" (So-So Technologies)
- Adopción inducida por costos regulatorios, no por eficiencia
- Implicaciones: desplazamiento sin ganancias de productividad

#### 2.3 Ondas Expansivas (Ripple Effects)
- Efectos indirectos sobre salarios y empleo en sectores adyacentes

#### 2.4 El Rol de la Informalidad
- La informalidad como amortiguador paradójico
- Modelo dual: sector formal automatizable vs. informal protegido por bajo costo

### 3. Análisis Estructural de Costos Laborales en Colombia
#### 3.1 Anatomía del Costo Total del Empleador
- Desglose: pensión (12%), salud (8.5%), ARL, prestaciones (21.83%), parafiscales (4-9%)
- Factor prestacional total: 46-58%
- Tabla comparativa con datos de Actualícese, ANIF, OECD

#### 3.2 Evolución del Salario Mínimo 2010-2026
- Crecimiento real de 40-45% en 15 años
- Incrementos Petro 2023-2026: +42% nominal en 3 años
- Comparación con productividad laboral (1-2% anual)

#### 3.3 La Paradoja de la Productividad
- Colombia: $16-21.6 USD/hora vs OCDE promedio
- Costos laborales = 45% de ingresos brutos vs 36% OCDE
- **EJERCICIO ECONOMÉTRICO 1:** Ratio costo laboral unitario / productividad por sector (datos EAM + Cuentas Nacionales)

#### 3.4 Comparación Internacional de Costos Laborales
- Tabla: compensación horaria manufactura, productividad/hora, densidad robots
- **EJERCICIO ECONOMÉTRICO 2:** Panel de países — relación costos laborales → densidad de robots

### 4. Brecha de Automatización: Colombia desde Casi Cero
#### 4.1 Densidad de Robots Industriales
- Colombia: 2-5 por 10,000 vs Corea del Sur: 1,012
- Datos IFR / Our World in Data

#### 4.2 Inversión en I+D
- Colombia: 0.30% PIB vs pares internacionales
- Datos World Bank WDI

#### 4.3 Dicotomía: Automatización Física vs Cognitiva
- Manufactura estancada vs servicios digitales acelerados
- Evidencia: Asobancaria (73% entidades con IA), BPO (80% con IA)

### 5. Evidencia Internacional: Costos Laborales como Catalizador
#### 5.1 Corea del Sur
- Incrementos salariales Moon Jae-in → aceleración robots
- Banco de Corea (2021): 1 robot/1000 trabajadores → -0.1pp empleo

#### 5.2 Alemania
- Reformas Hartz: trabajo barato → menos automatización
- Sistema dual de formación como amortiguador

#### 5.3 China
- Salarios crecientes → "Made in China 2025" → tercer lugar mundial en robots
- Subsidios provinciales masivos (Guangdong: $135B USD)

#### 5.4 Japón
- Automatización por escasez demográfica, no por costos
- Contraejemplo que confirma la regla

#### 5.5 México y Chile
- México: nearshoring + automatización greenfield
- Chile: automatización minera (BHP 100% autónoma)

#### **EJERCICIO ECONOMÉTRICO 3:** Panel internacional
- Modelo: `Robot_density = f(labor_cost, productivity, R&D/GDP, GDP_pc, demographics)`
- Países: 30-50 países, 2010-2023
- Datos: PWT 11.0 + ILOSTAT + WDI + IFR/OWID

### 6. Riesgo de Automatización en el Mercado Colombiano
#### 6.1 Proyecciones Agregadas
- Fedesarrollo: 57-58% empleos en riesgo
- Morales, Atis y Fajardo (2023): 37% en alto riesgo
- OECD enfoque tareas: 14% alto riesgo + 32% cambio profundo

#### 6.2 Vulnerabilidad Demográfica
- Jóvenes y mujeres con mayor exposición
- **EJERCICIO ECONOMÉTRICO 4:** P(riesgo_alto) = f(sector, formalidad, educación, sexo, edad, ingreso, tamaño_firma)
- Datos: GEIH + mapeo probabilidades Frey & Osborne a ocupaciones colombianas

#### 6.3 La Paradoja de la Informalidad
- Trabajadores informales: riesgo técnico alto, riesgo práctico bajo
- Empresas informales: sin capital para automatizar
- **EJERCICIO ECONOMÉTRICO 5:** Interacción formalidad × riesgo técnico → riesgo efectivo

#### 6.4 Impacto de IA Generativa
- OECD/GPAI "Voces del cambio": reverse skill bias
- Exposición por quintil de ingreso

### 7. Análisis Sectorial de Vulnerabilidad
#### 7.1 Construcción del Índice de Vulnerabilidad Compuesto

**EJERCICIO ECONOMÉTRICO 6 — El ejercicio central:**

Construir para cada sector j un **Índice de Vulnerabilidad a la Automatización (IVA)**:

```
IVA_j = w1 × PotencialTécnico_j × w2 × IncentivoCostoLaboral_j × w3 × Formalidad_j × w4 × MadurezTecnológica_j
```

Donde:
- `PotencialTécnico_j` = proporción de tareas automatizables (Frey & Osborne mapeado a CIIU)
- `IncentivoCostoLaboral_j` = ratio (costo laboral total / valor agregado por trabajador) del sector
- `Formalidad_j` = tasa de formalidad del sector (GEIH)
- `MadurezTecnológica_j` = proxy de adopción tecnológica actual (EDIT + indicadores de IA)

#### 7.2 Sectores de Mayor Riesgo
- Servicios financieros: alta formalidad, altos salarios, tecnología madura
- BPO y servicios empresariales: vulnerabilidad existencial
- Manufactura formal: automatización incipiente pero acelerándose
- Minería y energía formal: ya automatizando grandes operaciones

#### 7.3 Sectores de Menor Riesgo Práctico
- Agricultura (>85% informal)
- Construcción (70-75% informal)
- Hoteles y restaurantes (70-80% informal)

### 8. Política Pública y Marco Regulatorio
#### 8.1 Reforma Laboral (archivada marzo 2025)
- Incrementos proyectados: 18-34% (Fenalco)
- 17% de empresas declararía sustitución por tecnología
- Paradoja regulatoria: protección laboral que acelera automatización

#### 8.2 CONPES 4144 y Política Nacional de IA
- $479,000 millones COP hasta 2030
- SENATEC: 303,000 personas en TI (1.3% fuerza laboral)

#### 8.3 Desconexión Institucional
- Paradoja SENA: financiado por parafiscales que incentivan automatización

### 9. Proyecciones y Escenarios
#### **EJERCICIO ECONOMÉTRICO 7:** Simulación de escenarios
- Escenario base: costos laborales crecen al ritmo actual
- Escenario acelerado: reforma laboral aprobada
- Escenario moderado: reforma tributaria reduce parafiscales
- Variable de interés: empleos en riesgo efectivo por escenario

### 10. Conclusiones y Recomendaciones de Política

---

## Resumen de Ejercicios Econométricos

| # | Ejercicio | Datos | Método |
|---|-----------|-------|--------|
| 1 | Costo laboral unitario / productividad por sector | EAM + Cuentas Nacionales | Estadística descriptiva, ratios |
| 2 | Panel internacional: costos → densidad robots | PWT + ILOSTAT + WDI + IFR | Panel FE/RE, GMM |
| 3 | Panel de países: determinantes adopción robots | PWT + ILOSTAT + WDI + OWID | Panel dinámico, Arellano-Bond |
| 4 | Determinantes de riesgo de automatización individual | GEIH + Frey & Osborne | Logit/Probit, AME |
| 5 | Interacción formalidad × riesgo técnico | GEIH | Logit con interacciones |
| 6 | Índice de Vulnerabilidad Sectorial (IVA) | EAM + EDIT + GEIH + F&O | Índice compuesto, PCA |
| 7 | Simulación de escenarios | Todos | Microsimulación |

---

## Datos Requeridos por Ejercicio

### Ejercicio 1 (Descriptivo sectorial):
- [x] EAM 2021-2023 (DANE microdatos) — **requiere registro DANE**
- [x] Cuentas Nacionales productividad laboral — **descarga directa**
- [x] Cuentas Nacionales PIB sectorial — **descarga directa**

### Ejercicio 2-3 (Panel internacional):
- [x] Penn World Table 11.0 — **descarga directa**
- [x] ILOSTAT productividad por hora — **descarga directa**
- [x] World Bank WDI (I+D/PIB, PIB por trabajador) — **descarga directa**
- [x] Our World in Data robots instalados — **descarga directa**
- [ ] IFR robot density completo — **pagado EUR 1,850** (usar OWID + executive summaries)

### Ejercicio 4-5 (Riesgo individual Colombia):
- [x] GEIH 2023-2024 (DANE microdatos) — **requiere registro DANE**
- [x] Frey & Osborne probabilidades (Apéndice A del paper) — **extraer de PDF**
- [x] Tabla de concordancia SOC → ISCO → CIUO Colombia — **construir manualmente**

### Ejercicio 6 (Índice sectorial):
- [x] Todo lo anterior + EDIT X — **requiere registro DANE**

### Ejercicio 7 (Simulación):
- [x] Todo lo anterior
- [x] Parámetros de reforma laboral (ANIF, Fenalco)
