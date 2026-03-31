# Cognitive Factor Economics (CFE)
### *A Research Program on the Economics of Human-AI Complementarity in Society 5.0*

---

> **"The fundamental question of our era is not whether AI will replace human labor, but how the boundary between human and artificial cognition becomes itself a factor of production — and what economic theory has to say about it."**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Intellectual Genealogy](#2-intellectual-genealogy)
3. [The Core Theoretical Problem](#3-the-core-theoretical-problem)
4. [Foundational Model: Augmented Human Capital (AHC)](#4-foundational-model-augmented-human-capital-ahc)
5. [Methodological Framework: LLMs as Economic Measurement Instruments](#5-methodological-framework-llms-as-economic-measurement-instruments)
6. [Research Agenda — The Paper Map](#6-research-agenda--the-paper-map)
   - [Paper 0 — Methodology](#paper-0--llms-as-econometric-instruments-the-methodological-foundation)
   - [Paper 1 — Core Theory](#paper-1--augmented-human-capital-the-foundational-model)
   - [Paper 2 — Depreciation](#paper-2--cognitive-capital-depreciation)
   - [Paper 3 — Geography](#paper-3--geography-of-cognitive-augmentation)
   - [Paper 4 — Distributive Justice](#paper-4--distributive-justice-and-cognitive-capabilities)
   - [Paper 5 — Trade Theory](#paper-5--cognitive-comparative-advantage)
   - [Applied Line: Society 5.0 Workplace Transitions (Papers 6–9)](#applied-line-society-50-workplace-transitions-papers-69)
   - [Paper 6 — Workplace Augmentation Design](#paper-6--workplace-augmentation-design-for-society-50-first-investigation-in-this-line)
   - [Paper 7 — Competencies & Education](#paper-7--competencies-and-education-for-society-50-planned)
   - [Paper 8 — Sectoral S5.0 Models](#paper-8--sectoral-society-50-models-planned)
   - [Paper 9 — Governance & Sustainability](#paper-9--governance-sustainability-and-circular-economy-in-society-50-planned)
7. [Society 5.0 as Institutional Context](#7-society-50-as-institutional-context)
8. [Empirical Strategy — Data Architecture](#8-empirical-strategy--data-architecture)
9. [Positioning in the Literature](#9-positioning-in-the-literature)
10. [Key Hypotheses](#10-key-hypotheses)
11. [Contribution to Policy](#11-contribution-to-policy)
12. [Scope Conditions and Limitations](#12-scope-conditions-and-limitations)
13. [Glossary of CFE Concepts](#13-glossary-of-cfe-concepts)
14. [Citation Roadmap](#14-citation-roadmap)
15. [Author and Affiliation](#15-author-and-affiliation)

---

## 1. Overview

**Cognitive Factor Economics (CFE)** is a research program in economic theory and applied econometrics that addresses a fundamental gap in the economics literature: the absence of a formal, tractable, and empirically identified framework for modeling **cognitive production** in economies where artificial intelligence co-produces value alongside human workers.

The program proceeds from a single, precise observation:

> Standard economic theory — from the Solow growth model to Becker's human capital theory to Autor's task-based framework — treats cognition as a scalar attribute of labor. In the regime of generative AI, **cognition is a jointly produced output** of a human-machine system, and existing theory has no primitive for this.

CFE constructs that primitive, derives its implications for labor markets, geography, trade, institutional design, and distributive justice, and provides an empirically operational measurement methodology using Large Language Models (LLMs) as instruments for variables that previously could not be observed at scale.

The program is explicitly designed to:
- Produce **theoretically foundational** results, not conjunctural descriptions
- Be **atemporal**: the core models are valid regardless of which AI systems are deployed or when
- Be **empirically tractable**: every theoretical construct maps to a measurable variable
- Address **developing economies** — specifically Latin America — as primary empirical terrain, filling a systematic gap in the existing literature

---

## 2. Intellectual Genealogy

CFE does not emerge from a vacuum. It stands at the confluence of five established research traditions, inheriting their core questions while resolving tensions between them:

### 2.1 Human Capital Theory (Becker, 1964; Mincer, 1974)

The foundational insight: workers are heterogeneous in productivity not just because of physical endowments but because they invest in skills. CFE accepts this framework fully and asks: **what is the unit of human capital investment when AI can perform a growing subset of cognitive tasks?** The Mincerian equation — which regresses log wages on years of education and experience — is misspecified in CFE's framework, and the program produces a corrected specification.

### 2.2 Task-Based Models of Labor Markets (Autor, Levy & Murnane, 2003; Acemoglu & Restrepo, 2018–2022)

The key insight: technology does not substitute for workers, it substitutes for *tasks*. This decomposition is the most important methodological advance in labor economics in 30 years. CFE inherits the task-based architecture but argues it operates at the wrong level of aggregation for the generative AI era: **the relevant unit is not a task but a cognitive interaction** — a co-produced cognitive output of a human-AI dyad. The task framework is a special case of CFE where the augmentation function $\phi = 1$ (no complementarity, pure substitution or pure independence).

### 2.3 Growth Theory and General Purpose Technologies (Bresnahan & Trajtenberg, 1995; Helpman, 1998)

AI is widely discussed as a General Purpose Technology (GPT). The GPT literature explains why transformative technologies produce slow aggregate gains initially (the "productivity paradox") — because complementary investments are required to unlock their potential. CFE formalizes what those complementary investments are in the case of AI: they are investments in **augmentable human capital** ($H^A$), not generic education or infrastructure. This provides micro-foundations for the GPT productivity paradox specific to AI.

### 2.4 New Economic Geography (Krugman, 1991; Fujita, Krugman & Venables, 1999)

The NEG explains geographic concentration of production through increasing returns and transport costs. CFE extends this to the geography of *cognitive production*, where the relevant "transport costs" are not physical but institutional — regulatory distance, data governance, educational system quality. This generates predictions about why cognitive augmentation clusters in specific regions even when AI tools are digitally ubiquitous.

### 2.5 Capabilities Approach (Sen, 1999; Nussbaum, 2011)

Sen's framework evaluates well-being in terms of what people are able to *do* and *be*, not just what resources they hold. CFE introduces a new capability dimension — the **capability of augmentation** — and asks what a just distribution of AI access looks like from this perspective. This bridges economic theory, political philosophy, and AI governance.

---

## 3. The Core Theoretical Problem

### The Measurement Crisis of Cognitive Production

Current national accounts and microeconomic datasets face a fundamental measurement problem in the AI era:

**The AI Absorption Problem:** When a worker uses an AI assistant to produce a report, a legal brief, or a software module, the output is attributed entirely to the worker's labor in conventional measurement. The contribution of the AI system disappears into total factor productivity (TFP) as an unmeasured residual. This is not a minor measurement error — it is a structural misattribution that:

1. Biases estimates of labor productivity growth upward (workers appear more productive than they are)
2. Biases estimates of capital returns downward (AI capital contribution is unmeasured)
3. Makes the elasticity of substitution between AI and human labor unidentified
4. Renders standard wage equations misspecified in a theoretically predictable way

**The Heterogeneity Problem:** Even within the category of "cognitive work," the literature has established that AI is simultaneously:
- A near-perfect *substitute* for some cognitive tasks (routine information processing, pattern recognition over structured data)
- A genuine *complement* for others (complex judgment, contextual interpretation, novel synthesis)
- Largely *irrelevant* for physical-manual tasks regardless of cognitive content

Standard indices — Felten et al. (2021), Webb (2020), Eloundou et al. (2023) — measure aggregate exposure but do not distinguish between substitution and complementarity at the task level. This is the operationalization gap that CFE fills.

**The Development Gap:** The overwhelming majority of empirical work on AI and labor markets uses data from the United States or Western Europe. Developing economies — with their distinct characteristics of labor market informality, weaker institutional complementarity, different educational systems, and different sectoral composition — are either absent or treated as residual cases. This is not just a gap in coverage; it is a theoretical problem, because the mechanisms by which AI affects labor markets differ fundamentally when a large fraction of the workforce is informal.

---

## 4. Foundational Model: Augmented Human Capital (AHC)

The central theoretical contribution of CFE is the **Augmented Human Capital model**, which provides the micro-foundation for all subsequent papers.

### 4.1 Decomposition of Human Capital

The AHC model decomposes individual human capital $H_i$ into three orthogonal components:

$$H_i = H_i^P \oplus H_i^C \oplus H_i^A$$

Where:

| Component | Symbol | Definition | AI Relationship |
|-----------|--------|------------|----------------|
| Physical-Manual Capital | $H_i^P$ | Embodied skills requiring physical presence and motor coordination | Substitutable by robotics; largely unaffected by GenAI |
| Routine Cognitive Capital | $H_i^C$ | Cognitive skills for structured, repeatable information tasks | Substitutable by GenAI — the displaced component |
| Augmentable Cognitive Capital | $H_i^A$ | Cognitive skills that GenAI *amplifies* rather than replaces: judgment, contextual reasoning, novel synthesis, ethical evaluation | Complementary to GenAI — the premium component |

This decomposition is theoretically motivated, not ad hoc. It maps directly to the task-based framework:
- $H^P$ corresponds to tasks with low AI exposure by any measure
- $H^C$ corresponds to tasks with high AI exposure and high substitutability ($\sigma > 1$)
- $H^A$ corresponds to tasks with high AI exposure but high complementarity ($\sigma < 1$ in interaction with AI capital)

### 4.2 The Production Function

At the firm level, output is produced by a hardware-software structure with three sectors:

$$Y_f = F\left(K_f^{HW},\; \underbrace{L_f^P \cdot H_f^P + \kappa K_f^{Rob}}_{\text{hardware sector}},\; \underbrace{L_f^C \cdot H_f^C + \phi(D_f) \cdot L_f^A \cdot H_f^A \cdot D_f}_{\text{software sector}}\right)$$

Where:
- $K_f^{HW}$ is physical capital (hardware)
- $K_f^{Rob}$ is robotic capital, substitutable with physical labor
- $D_f$ is the firm's stock of digital labor (AI systems)
- $\phi(D_f)$ is the **augmentation function**: increasing, concave, with $\phi(0)=1$ and $\lim_{D \to \infty} \phi(D) = \bar{\phi} < \infty$

The augmentation function $\phi$ is the key new object in CFE. It captures the productivity multiplier that AI provides to workers with augmentable capital $H^A$. Its properties — shape, curvature, upper bound $\bar{\phi}$ — are empirically estimable and vary by occupation, sector, and institutional context.

### 4.3 The Corrected Wage Equation

The standard Mincerian log-wage equation is:

$$\ln w_i = \alpha + \beta_1 \cdot Educ_i + \beta_2 \cdot Exp_i + \varepsilon_i$$

The CFE correction is:

$$\ln w_{iot} = \alpha + \underbrace{\beta_1 \cdot H_o^A + \beta_2 \cdot H_o^C}_{\text{direct capital returns}} + \underbrace{\beta_3 \cdot H_o^A \times \ln D_{f(i)t} + \beta_4 \cdot H_o^C \times \ln D_{f(i)t}}_{\text{augmentation interactions}} + \gamma X_{it} + \mu_i + \delta_t + \varepsilon_{iot}$$

**The CFE signature prediction:** $\beta_3 > 0$ and $\beta_4 < 0$. The return to augmentable capital increases with AI adoption; the return to routine cognitive capital decreases. This is not the standard automation story (which predicts $\beta_4 < 0$ only) — it predicts a simultaneous *premium* and *penalty* within the same education level, depending on the composition of human capital.

### 4.4 The Augmentation Threshold

A central result is the existence of an **augmentation threshold** $D^*$ such that:

- For $D_f < D^*$: AI adoption reduces average wages (substitution dominates)
- For $D_f > D^*$: AI adoption increases wages for $H^A$-intensive workers (complementarity dominates)

The threshold is determined by the structural parameters of the production function and provides a micro-foundation for heterogeneous findings in the empirical literature — some studies find AI raises wages, others find it depresses them. CFE reconciles these findings: both are correct, for different ranges of $D$.

---

## 5. Methodological Framework: LLMs as Economic Measurement Instruments

### 5.1 The Measurement Challenge

The central empirical challenge in CFE is that $H_i^A$ — augmentable cognitive capital — is not directly observable in any existing dataset. Survey-based measures of skills (O\*NET, ESCO, PIAAC) predate the generative AI era and do not distinguish between $H^C$ and $H^A$. Standard cognitive test scores measure total cognitive ability, not its augmentability.

CFE proposes a solution: **LLMs as measurement instruments for latent economic variables**.

### 5.2 The LLM Measurement Pipeline

The $AHC_o$ index for occupation $o$ is constructed as follows:

**Step 1 — Task decomposition:** Each occupation in O\*NET/ESCO is decomposed into its constituent tasks $\{t_k\}_{k=1}^K$ with importance weights $\{w_k\}$.

**Step 2 — LLM annotation:** A fine-tuned LLM evaluates each task on two dimensions:
```
For each task t_k:
  a_k = P(task is augmentable by GenAI | full task description)
         × P(AI acts as complement, not substitute | task context)

  Range: a_k ∈ [0, 1]
  0 = pure substitution / pure irrelevance
  1 = pure complementarity / maximum augmentation potential
```

**Step 3 — Index construction:**
$$AHC_o = \sum_{k=1}^K w_k \cdot a_{ok}, \quad AHC_o \in [0,1]$$

**Step 4 — Dual index:** A parallel index captures routine cognitive exposure:
$$H_o^C = \sum_{k=1}^K w_k \cdot s_{ok}$$

Where $s_{ok} = P(\text{task is substitutable by GenAI})$. By construction, $a_{ok}$ and $s_{ok}$ are not complements of each other — a task can be neither augmentable nor substitutable (physical tasks), or substitutable without being augmentable.

### 5.3 Validation Strategy

The LLM-generated indices are validated against three criteria:

**Convergent validity:** $AHC_o$ should correlate positively with existing AI exposure indices (Felten 2021, Webb 2020) for the component those indices capture correctly, while diverging for tasks where the distinction between augmentation and substitution matters.

**Discriminant validity (the hard test):** $AHC_o$ and $H_o^C$ should be orthogonal after controlling for general cognitive demand. If the LLM merely recovers "cognitive intensity," the two indices will be highly correlated. Orthogonality is the signature of genuine conceptual distinction.

**Predictive validity:** $AHC_o$ should predict:
- Post-ChatGPT wage premiums (positive, for augmentable occupations)
- Post-ChatGPT employment changes (positive for $AHC_o$, negative for $H_o^C$)
- Demand for AI-augmented skills in job postings (positive relationship)

### 5.4 Why LLMs Are Valid Instruments Here — The Econometric Argument

A concern with LLM-generated indices is endogeneity: if the LLM's training data reflects current wage structures, the index may mechanically predict wages. CFE addresses this through three strategies:

1. **Semantic vs. revealed preference:** The annotation prompt is strictly about task content and AI capability, not about wages or employment. The LLM evaluates task descriptions, not labor market outcomes.

2. **Cross-validation with pre-AI data:** The index is validated by predicting *pre-ChatGPT* skill premiums — if it predicts 2015–2022 wage patterns, it is capturing something real about task content, not AI-era labor market equilibria.

3. **Jackknife robustness:** The index is constructed independently with different LLM families (Llama 3, Mistral, GPT-4) and the intersection of high-confidence annotations forms the core index. High variance across models flags tasks where classification is genuinely uncertain.

---

## 6. Research Agenda — The Paper Map

The CFE program consists of ten papers organized in two tracks. **Core Track** (Papers 0–5): Paper 0 is the methodological foundation; Papers 1–5 are substantive theoretical and empirical contributions. **Applied Track** (Papers 6–9): the Society 5.0 Workplace Transitions line, which operationalizes CFE's primitives for institutional design, education, sectoral implementation, and governance. Each paper stands alone but collectively they form a unified program.

---

### Paper 0 — LLMs as Econometric Instruments: The Methodological Foundation

**Full title:** *"Measuring What Cannot Be Surveyed: LLMs as Instruments for Latent Cognitive Variables in Labor Economics"*

**Target journals:** *Journal of Econometrics* · *Econometrica* (short paper) · *Journal of Economic Methodology*

**Abstract (working):** This paper establishes the theoretical and practical foundations for using Large Language Models as measurement instruments for latent economic variables — specifically variables that describe the cognitive content of occupational tasks at a level of granularity not achievable with existing survey instruments. We formalize the conditions under which LLM-generated scores constitute valid instruments: exogeneity (the scoring process is independent of the outcome variable), relevance (the scores predict the latent variable), and monotonicity (higher scores correspond to higher latent values in a theoretically consistent direction). We then apply this framework to construct the *Augmentable Human Capital Index* ($AHC_o$) for all occupations in O\*NET and ESCO, validate it against multiple criteria, and demonstrate its superior predictive power relative to existing AI exposure measures for post-2022 labor market outcomes.

**Key contribution:** This paper is not about AI and labor markets per se — it is about measurement theory in economics, with a specific application. The methodology is generalizable to any domain where institutional quality, policy content, contract terms, or other semantic objects need to be quantified at scale.

---

### Paper 1 — Augmented Human Capital: The Foundational Model

**Full title:** *"Augmented Human Capital: A Unified Theory and LLM-Based Measurement Framework for Cognitive Factor Decomposition in AI-Augmented Economies"*

**Target journals:** *Journal of Political Economy* · *Review of Economic Studies* · *American Economic Review*

**Abstract (working):** We propose a decomposition of human capital into three components — physical-manual, routine cognitive, and augmentable cognitive — and develop a production function in which AI capital ($D$) interacts with these components asymmetrically: substituting for routine cognitive work while complementing augmentable cognitive work via an augmentation function $\phi(D)$. We derive the corrected Mincerian wage equation implied by this model and show that the standard specification is misspecified in a theoretically predictable direction. Using LLM-generated measures of $H^A$ and $H^C$ for all O\*NET occupations, merged with longitudinal household survey data from Colombia (GEIH 2018–2025), Mexico (ENOE), Chile (CASEN), and Brazil (PNAD), we estimate the model via instrumental variables. We find strong support for the signature prediction: the wage return to $H^A$ increases with firm-level AI adoption, while the return to $H^C$ decreases, with the crossover occurring at an augmentation threshold $D^*$ that varies by sector. We discuss implications for the theory of human capital and for education policy.

**Key contribution:** Extends Becker (1964) and Mincer (1974) with a theoretically grounded decomposition of cognitive capital. Provides the first causal estimates of the augmentation premium $\phi$ for Latin American labor markets. Introduces the $AHC_o$ index as a new standard instrument for labor economics research in the AI era.

**Data:** GEIH (DANE, Colombia), ENOE (INEGI, Mexico), CASEN (INE, Chile), PNAD (IBGE, Brazil), Revelio Labs (firm-level AI adoption), O\*NET task database.

**Identification:** IV with sectoral AI patent intensity (lagged 3 years) as instrument for firm-level AI adoption $D_f$.

---

### Paper 2 — Cognitive Capital Depreciation

**Full title:** *"Cognitive Capital Depreciation in the Age of Generative AI: A Dynamic Human Capital Model with Endogenous Skill Half-Lives"*

**Target journals:** *Journal of Economic Theory* · *Review of Economics and Statistics* · *Journal of Human Capital*

**Abstract (working):** Standard human capital theory treats the depreciation rate $\delta_H$ as a constant biological parameter. We argue this is misspecified in the AI era: when AI advances, cognitive skills become economically obsolete at a rate determined by the speed of technological change rather than time. We develop a dynamic model in which $\delta_H(t) = \delta_0 + \lambda \cdot \dot{\Omega}_{AI}(t)$, where $\Omega_{AI}(t)$ is the set of cognitive tasks performable by AI at time $t$ and $\lambda$ captures the transmission speed from technological frontier to labor market. Using optimal control theory, we derive the worker's optimal investment path in augmentable vs. routine cognitive capital under this endogenous depreciation regime, and show that there exists a technological velocity threshold $\dot{\Omega}^*$ above which investment in routine cognitive capital has negative net present value. We operationalize $\dot{\Omega}_{AI}(t)$ using temporal embeddings of AI capability benchmarks (MMLU, HumanEval, and task-specific evaluations) processed through our LLM measurement pipeline. Empirical estimation uses the differential timing of AI capability advances across occupational categories as a natural experiment.

**Key contribution:** Provides the first endogenous theory of cognitive capital depreciation. Resolves the puzzle of why education returns for routine cognitive occupations have declined faster than standard depreciation models predict. Has direct implications for optimal educational investment horizons and reskilling policy design.

**Key concept — Skill Half-Life:** The *cognitive skill half-life* $t_{1/2}$ is the time after which a skill's economic value has depreciated to 50% of its initial level. Under standard theory, $t_{1/2}$ is measured in decades. Under the CFE model, for high-$\dot{\Omega}_{AI}$ occupations, $t_{1/2}$ can fall to 3–7 years, fundamentally changing the economics of educational investment.

---

### Paper 3 — Geography of Cognitive Augmentation

**Full title:** *"Cognitive Augmentation Clusters: A New Economic Geography of Human-AI Complementarity"*

**Target journals:** *Journal of Urban Economics* · *Regional Science and Urban Economics* · *Economic Geography*

**Abstract (working):** Why does the capacity for AI-augmented cognitive production cluster geographically despite AI tools being digitally ubiquitous? We propose a New Economic Geography model where the centripetal force for cognitive production clusters is not physical transport costs (Krugman, 1991) but institutional complementarity distance: the gap between a region's institutional profile (data governance, educational system, regulatory quality, AI infrastructure) and the threshold required for productive augmentation. We show this model generates multiple equilibria: regions above the institutional complementarity threshold attract augmentation-intensive activity, while regions below are trapped in a low-augmentation equilibrium regardless of their physical proximity to technology. We estimate the model using AI adoption data at the departmental level for Colombia and at the NUTS-2 level for Europe, exploiting pre-existing variation in university STEM density as an instrument for institutional complementarity capacity. Results confirm the multiple-equilibria prediction: the distribution of cognitive augmentation is bimodal, not continuous, consistent with a threshold process.

**Key contribution:** Provides a formal theory of why AI adoption is geographically concentrated even without physical barriers. Introduces the concept of *institutional complementarity distance* as a determinant of AI diffusion. Has direct implications for regional development policy and the design of "AI-ready" institutional environments in Society 5.0.

---

### Paper 4 — Distributive Justice and Cognitive Capabilities

**Full title:** *"Distributive Justice in AI-Augmented Economies: A Capabilities Approach to Cognitive Factor Allocation"*

**Target journals:** *Journal of Public Economics* · *Economics and Philosophy* · *World Development*

**Abstract (working):** The adoption of AI as a factor of production creates a new form of inequality: *augmentation inequality* — the gap between individuals with and without effective access to AI-augmented cognitive production — that is orthogonal to standard measures of income, wealth, or educational inequality. We extend Sen's capability approach to include the *capability of augmentation* $C_i^A$ — the effective ability to amplify cognitive output through AI — as a fundamental dimension of human freedom and economic opportunity. We then address the distributional question formally: using a Rawlsian veil-of-ignorance argument, we derive the optimal social contract for AI access distribution and show that it implies universal access to a minimum level of augmentation capacity $D^{min}$, financed by a progressive tax on augmentation rents. We operationalize augmentation inequality using our $AHC_o$ index merged with household survey data and digital access indicators for five Latin American countries, producing the first cross-country measurement of augmentation inequality. We find that augmentation inequality is large, growing, and only partially correlated with income inequality.

**Key contribution:** First formal integration of the capabilities approach with AI economics. Introduces *augmentation inequality* as a measurable economic concept. Derives the welfare-maximizing AI access policy from first principles of distributive justice. Provides empirical benchmarks for augmentation inequality in Latin America.

---

### Paper 5 — Cognitive Comparative Advantage

**Full title:** *"Cognitive Comparative Advantage: AI Augmentation, Human Capital Endowments, and the New Basis for Trade in Knowledge Services"*

**Target journals:** *Journal of International Economics* · *American Economic Journal: Macroeconomics* · *Review of International Economics*

**Abstract (working):** The Heckscher-Ohlin theorem predicts that countries export goods intensive in their abundant factors. Applied to knowledge services, this implies that countries with abundant educated labor should export cognitive services. We show this prediction is misspecified in the AI era: what matters for comparative advantage in AI-augmented knowledge services is not the *quantity* of educated labor but the *composition* of human capital — specifically the ratio $H^A/H^C$ — and the country's AI access $D^{access}$. We develop a two-country, two-sector model with tradeable cognitive services and show that: (i) a country with abundant educated labor but low $H^A/H^C$ ratio can *lose* comparative advantage in premium cognitive services to a country with less overall education but better composition; (ii) AI access asymmetries between countries generate a new margin of trade that can either reinforce or reverse traditional comparative advantage; and (iii) the offshoring of cognitive services may partially reverse as AI augmentation in high-wage countries reduces the cost advantage of low-wage countries for routine cognitive work. We calibrate the model using trade in services data (WTO/IMF BOP) and our $AHC_o$ indices for 40 countries, and test its predictions against observed shifts in knowledge service trade flows post-2022.

**Key contribution:** First general-equilibrium trade model incorporating AI augmentation as a determinant of comparative advantage. Explains the empirically observed patterns of knowledge service trade realignment post-ChatGPT. Has implications for development strategy in countries whose growth model depends on cognitive service exports (India, Philippines, Colombia).

---

### Applied Line: Society 5.0 Workplace Transitions (Papers 6–9)

The theoretical core of CFE (Papers 0–5) establishes the primitives — $H^A$, $\phi(D)$, $D^*$, $\tau_{ij}$ — but leaves open the question of *how to operationalize* these concepts in the design of actual institutions, workplaces, and educational systems. The **Society 5.0 Applied Line** extends CFE into implementation science: how organizations and societies transition from automation-centered (Industry 4.0) to augmentation-centered (Society 5.0) paradigms.

This line addresses four interconnected research questions:

| Paper | Core Question | CFE Primitive Extended |
|-------|--------------|----------------------|
| **Paper 6** — Workplace Augmentation Design | How should work environments be designed to maximize $\phi(D)$? | Endogenizes $\phi(D) \to \phi(D, W)$ where $W$ = workplace design |
| **Paper 7** — Education & Competencies for S5.0 | What competencies must educational systems develop for Society 5.0? | Operationalizes $H^A$ formation: curriculum → $\Delta H^A$ pipeline |
| **Paper 8** — Sectoral S5.0 Models (Health, Smart Universities) | How do sector-specific constraints reshape augmentation? | Estimates sector-specific $\phi_s(D, W)$ and risk profiles |
| **Paper 9** — Governance, Sustainability & Circular Economy in S5.0 | What governance structures sustain human-centric AI transitions? | Formalizes $\tau_{ij}$ as governance design variable |

---

### Paper 6 — Workplace Augmentation Design for Society 5.0 *(First investigation in this line)*

**Full title:** *"From Automation to Augmentation: A Framework for Designing Human-Centric Work Environments in Society 5.0"*

**Target journals:** *Technological Forecasting and Social Change* (Q1, IF ≈ 12) · *Computers in Industry* · *Technology in Society*

**Abstract (working):** The transition from Industry 4.0 to Society 5.0 requires a fundamental shift in workplace design: from optimizing automation (minimizing human labor per unit of output) to optimizing augmentation (maximizing human-AI complementarity per unit of cognitive output). Yet no formal framework exists for what constitutes "human-centric" workplace design in operational, measurable terms. We address this gap by endogenizing the augmentation function from the Cognitive Factor Economics framework: instead of treating $\phi(D)$ as exogenous, we model it as $\phi(D, W)$, where $W$ is a vector of workplace design parameters encompassing five dimensions: (i) AI interface design and usability, (ii) human-AI decision authority allocation, (iii) task granularity and workflow orchestration, (iv) feedback and learning loop architecture, and (v) psychosocial work environment (autonomy, meaning, wellbeing). We define *human-centricity* formally as the set of design choices $W^*$ that satisfy $\partial \phi / \partial W > 0$ — a design is human-centric if and only if it increases the augmentation multiplier. We develop the **Workplace Augmentation Design Index (WADI)**, a composite instrument that measures the distance between a firm's current workplace design and the augmentation-optimal design $W^*$, conditional on its workforce's $H^A$ composition. Using mixed methods — a primary survey of 200+ Colombian firms across manufacturing, services, BPO, and financial sectors, merged with DANE EDIT technology investment data and our $AHC_o$ indices — we estimate the relationship between WADI scores and effective augmentation. We find that: (a) workplace design parameters explain 35–40% of the variance in effective $\phi$ after controlling for AI investment $D$ and worker composition; (b) the most binding constraint for Society 5.0 transitions is not technology adoption but decision authority allocation — firms that centralize AI-mediated decisions in management rather than distributing them to augmented workers achieve significantly lower $\phi$; (c) the optimal workplace design is contingent on the $H^A/H^C$ composition of the workforce, generating a *design-composition complementarity* that creates path dependence in organizational transitions. We derive a transition roadmap with three phases — diagnostic (measure WADI), redesign (optimize $W$ given $H^A$ composition), and institutional embedding (governance structures for sustained human-centricity) — and discuss implications for firms, labor policy, and the Society 5.0 agenda.

**Key contributions:**

1. **Theoretical:** Endogenizes the augmentation function $\phi(D) \to \phi(D, W)$. Provides the first formal definition of "human-centricity" grounded in economic theory rather than normative aspiration. Shows that human-centric design is not altruism — it is the profit-maximizing strategy when $H^A$ is the scarce factor.

2. **Methodological:** Introduces the **WADI** (Workplace Augmentation Design Index) — a validated, multi-dimensional instrument that makes Society 5.0 transition progress measurable at the firm level. Analogous to what the $AHC_o$ index does for occupations, WADI does for workplaces.

3. **Empirical:** First developing-country evidence on the relationship between workplace design and AI augmentation effectiveness. Identifies *decision authority allocation* as the primary bottleneck — not technology access, not skills, but organizational power structures.

4. **Policy:** Produces a concrete transition roadmap (diagnostic → redesign → embedding) usable by firms and policymakers implementing Society 5.0 strategies.

**Data:**

| Source | Coverage | Variables |
|--------|----------|-----------|
| Primary survey (WADI instrument) | 200+ Colombian firms, 4 sectors | 5-dimension workplace design scores |
| DANE EDIT X–XI | Manufacturing firms | Technology investment, innovation types |
| DANE EMICRON | Micro-firm technology adoption | Digital tool usage, productivity |
| $AHC_o$ index (from Paper 1) | All occupations mapped to CIUO-08 | Augmentation potential per occupation |
| Computrabajo/LinkedIn | Job postings, 2022–2026 | AI skill requirements, job redesign signals |
| DANE GEIH | Household surveys | Worker characteristics, wages, formality |

**Identification strategy:** Cross-sectional OLS + IV. The WADI score is instrumented with pre-COVID organizational structure variables (management layers, worker autonomy indices from DANE Encuesta de Micronegocios 2019) to address the reverse causality that more productive firms may both adopt better designs and achieve higher $\phi$.

**The five WADI dimensions:**

| Dimension | Measures | Theoretical Basis |
|-----------|----------|-------------------|
| **W₁ — AI Interface Design** | Usability, transparency, explainability of AI tools deployed | HCI literature + augmented reality decision-making (Endsley 2017) |
| **W₂ — Decision Authority Allocation** | Who decides: human, AI, or human-with-AI? At what level? | Organizational economics (Aghion & Tirole 1997) + S5.0 human-centricity |
| **W₃ — Task Orchestration** | Granularity of human-AI task division; workflow integration | Task-based models (Autor 2003) + process mining |
| **W₄ — Learning Loop Architecture** | Feedback cycles: does the AI learn from human corrections? Does the human learn from AI suggestions? | Dynamic capabilities (Teece 2018) + human-in-the-loop ML |
| **W₅ — Psychosocial Environment** | Autonomy, meaning, cognitive load, wellbeing in AI-augmented work | Karasek demand-control model + Society 5.0 wellbeing goals |

**Connection to subsequent papers:**

- Paper 7 (Education) uses WADI to identify which competencies ($H^A$ sub-components) are most demanded by high-WADI workplaces
- Paper 8 (Sectoral) estimates sector-specific $\phi_s(D, W)$ for health and smart universities
- Paper 9 (Governance) uses WADI adoption patterns to identify governance structures that accelerate or block S5.0 transitions

---

### Paper 7 — Competencies and Education for Society 5.0 *(Planned)*

**Full title:** *"Augmentable by Design: New Competency Frameworks for Education in Society 5.0"*

**Target journals:** *Computers & Education* · *Higher Education* · *Studies in Higher Education*

**Core question:** What specific competencies must educational systems develop to maximize $H^A$ — and how do these differ from traditional STEM or "21st century skills" frameworks? Includes sub-questions on smart university risks, AR/VR-mediated decision-making in educational settings, and curricular redesign for augmentation readiness.

---

### Paper 8 — Sectoral Society 5.0 Models *(Planned)*

**Full title:** *"Society 5.0 in Practice: Sector-Specific Augmentation Models for Health, Education, and Knowledge Services"*

**Target journals:** *Technological Forecasting and Social Change* · *Government Information Quarterly* · *Health Policy and Technology*

**Core question:** How do sector-specific constraints (regulation, risk tolerance, human interaction intensity) reshape the augmentation function? Develops sector-specific $\phi_s(D, W)$ models for healthcare (patient safety constraints, clinical judgment augmentation) and higher education (smart university risks, pedagogical augmentation vs. substitution).

---

### Paper 9 — Governance, Sustainability and Circular Economy in Society 5.0 *(Planned)*

**Full title:** *"Governing the Augmented Economy: Institutional Design for Sustainable Human-AI Complementarity"*

**Target journals:** *Research Policy* · *Ecological Economics* · *Regulation & Governance*

**Core question:** What governance structures sustain human-centric AI transitions — and how does the circular economy paradigm interact with cognitive augmentation? Formalizes the institutional complementarity distance $\tau_{ij}$ as a governance design variable and models the sustainability constraints on Society 5.0 transitions (energy costs of AI, circular economy principles for cognitive infrastructure).

---

## 7. Society 5.0 as Institutional Context

CFE is framed within the broader transition to **Society 5.0** — the concept articulated by the Japanese Cabinet Office (2016) and developed by the European Commission (2021) as the successor paradigm to Industry 4.0. Where Industry 4.0 focused on automation of physical and routine cognitive processes, Society 5.0 focuses on **human-centricity** in an AI-saturated world: technology designed to serve human flourishing rather than to maximize throughput.

CFE provides the **economic foundations** for Society 5.0 in three ways:

### 7.1 The Economic Definition of Human-Centricity

Society 5.0 calls for "human-centric" technology but lacks a formal economic definition. CFE proposes: a human-centric technology allocation is one that maximizes investment in $H^A$ — augmentable human capital — rather than one that merely minimizes displacement of $H^C$. This shifts the policy question from "how do we protect jobs?" to "how do we maximize augmentation?"

### 7.2 The Institutional Requirements for Society 5.0

CFE's geography paper (Paper 3) shows that Society 5.0 outcomes require institutional complementarity above a threshold. The program identifies the specific institutional components:

- **Data governance infrastructure:** Enables AI systems to be trained on locally relevant data
- **Educational system alignment:** Schools and universities that develop $H^A$ rather than $H^C$
- **Labor market flexibility:** Allows workers to transition from $H^C$-intensive to $H^A$-intensive roles
- **AI access policy:** Ensures the augmentation capability is not monopolized by large firms

### 7.3 The Welfare Criterion for Society 5.0 Transitions

CFE's justice paper (Paper 4) provides a welfare criterion for evaluating Society 5.0 transitions: progress occurs when augmentation inequality decreases while average augmentation capability increases. This is analogous to the Atkinson inequality-aversion criterion but applied to the distribution of cognitive augmentation rather than income.

---

## 8. Empirical Strategy — Data Architecture

The empirical program relies on a layered data architecture:

### Layer 1 — Occupational Task Data
| Source | Coverage | Variables |
|--------|----------|-----------|
| O\*NET (US DOL) | ~1,000 occupations, 35 task dimensions | Task importance, frequency, skill requirements |
| ESCO (EU Commission) | ~3,000 occupations, multilingual | Task descriptions, skill taxonomies |
| CEPAL SISCO | Latin American occupational classifications | ALC-specific task content |

### Layer 2 — LLM-Generated Indices
| Index | Construction | Validation |
|-------|-------------|------------|
| $AHC_o$ | LLM annotation of augmentation potential | Cohen's κ, predictive validity |
| $H_o^C$ | LLM annotation of substitution exposure | Correlation with Felten (2021), Webb (2020) |
| $\dot{\Omega}_{AI}(t)$ | Temporal embedding of AI benchmarks | Benchmark-capability panel |

### Layer 3 — Household Survey Microdata
| Country | Survey | Sample | Period |
|---------|--------|--------|--------|
| Colombia | GEIH (DANE) | ~200K obs/year | 2018–2025 |
| Mexico | ENOE (INEGI) | ~300K obs/quarter | 2018–2025 |
| Chile | CASEN (INE) | ~200K obs biennial | 2017–2024 |
| Brazil | PNAD Contínua (IBGE) | ~500K obs/quarter | 2018–2025 |
| Regional | EU-LFS, CPS (US) | — | 2018–2025 |

### Layer 4 — Firm-Level AI Adoption
| Source | Coverage | Variables |
|--------|----------|-----------|
| Revelio Labs | Firm-occupation level, US+ALC | AI skill demand, workforce composition |
| Computrabajo / LinkedIn | Job postings, ALC | Skill requirements, wages offered |
| DANE EDIT | Colombian manufacturing firms | Technology investment |
| Orbis-BvD | Multinational firms | AI patent count, R&D |

### Layer 5 — Institutional and Macro Data
| Source | Variables |
|--------|-----------|
| World Bank WGI | Governance quality indicators |
| OECD Digital Economy Outlook | Broadband, AI adoption rates |
| IMF AI Preparedness Index | Country-level AI readiness |
| WIPO Patent Database | AI patent counts by sector and country |

---

## 9. Positioning in the Literature

### What CFE is NOT

- It is **not** an automation-displacement paper in the Frey-Osborne tradition. Those papers estimate which jobs are at risk; CFE theorizes *why* and derives welfare implications.
- It is **not** an AI enthusiasm paper predicting productivity booms. CFE is agnostic on aggregate effects and focuses on distribution and composition.
- It is **not** a conjunctural analysis of current AI systems. The framework is designed to be valid for any AI system that exhibits cognitive complementarity, regardless of which specific models are deployed.

### How CFE Advances Each Strand

| Literature Strand | Current Frontier | CFE Contribution |
|-------------------|-----------------|------------------|
| Human Capital Theory | Returns to education declining for routine cognitive work (Autor 2019) | Formal decomposition of *why* via $H^A/H^C$ distinction; corrected wage equation |
| Task-Based Models | Displacement vs. reinstatement margins (Acemoglu & Restrepo 2022) | Introduces augmentation function $\phi(D)$ as third margin; new sufficient statistic |
| AI Measurement | Exposure indices (Felten 2021, Eloundou 2023) | First validated augmentability index distinguishing substitution from complementarity |
| New Economic Geography | AI adoption clusters (OECD 2025) | Formal model with institutional complementarity threshold; multiple equilibria |
| Development Economics | AI and developing countries (Cazzaniga IMF 2024) | ALC-specific empirics; informality as distinct transmission mechanism |
| Trade Theory | AI and trade (WTO 2025) | New basis of comparative advantage in cognitive services via $H^A/H^C$ composition |
| Political Philosophy | AI and justice (Rawls applications) | First formal integration with Sen's capabilities; derives optimal AI access policy |

---

## 10. Key Hypotheses

The following hypotheses are testable with available data and constitute the empirical spine of the CFE program:

**H1 — The Augmentation Premium:** Workers in occupations with high $AHC_o$ receive a wage premium that increases with firm-level AI adoption $D_f$, after controlling for educational attainment.

**H2 — The Routine Cognitive Penalty:** Workers in occupations with high $H_o^C$ (routine cognitive) receive a wage penalty that increases with firm-level AI adoption, after controlling for educational attainment.

**H3 — The Within-Education Crossover:** The augmentation premium and the routine cognitive penalty exist *within* educational categories — specifically within the college-educated workforce — generating intra-educational wage polarization not captured by standard models.

**H4 — The Threshold Effect:** The relationship between AI adoption and wages is non-monotone: below the augmentation threshold $D^*$, AI adoption depresses wages on average; above $D^*$, AI adoption increases wages for $H^A$-intensive workers.

**H5 — Endogenous Depreciation:** The economic value of routine cognitive skills depreciates at a rate that is positively correlated with the rate of AI capability advancement in the corresponding task domain.

**H6 — The Institutional Complementarity Threshold:** Regions below a minimum institutional complementarity threshold fail to achieve cognitive augmentation clusters, regardless of their proximity to AI technology or educational investment.

**H7 — The Augmentation Inequality Gradient:** Augmentation inequality is larger in countries with high income inequality, weak data governance, and low educational system alignment — but this relationship is non-linear, with a threshold below which even high income inequality does not predict high augmentation inequality (because AI adoption is too low for the mechanism to activate).

**H8 — Cognitive Comparative Advantage:** Post-2022 shifts in knowledge service trade flows are better predicted by changes in $H^A/H^C$ composition than by changes in aggregate educational attainment or wage costs.

---

## 11. Contribution to Policy

While CFE is a theoretical and empirical program, its results have direct implications for policy in each of the substantive domains it addresses:

### Education Policy

The Skill Half-Life result (Paper 2) implies that the optimal educational investment changes fundamentally:
- Investment in $H^C$ (routine cognitive skills: structured analysis, data processing, standard coding) should be treated as short-horizon investment with high depreciation
- Investment in $H^A$ (augmentable skills: contextual judgment, cross-domain synthesis, ethical reasoning, relational intelligence) should be prioritized for long-horizon returns
- Curriculum design should be evaluated not just by current labor market returns but by the *trajectory* of returns as AI adoption advances

**Policy Implication:** Educational systems that have historically been evaluated by graduate employment rates need a new metric — the *cognitive augmentation readiness* of graduates, measurable with our $AHC$ framework.

### Labor Market Policy

The Augmentation Threshold result (Paper 1) implies that:
- Policies that restrict AI adoption (to protect $H^C$ workers) below the threshold increase welfare by preventing wage depression; above the threshold they reduce welfare by blocking augmentation gains
- Active labor market policies should be targeted specifically at transitions from $H^C$-intensive to $H^A$-intensive roles, not at protecting existing jobs

### Regional Development Policy

The Geography result (Paper 3) implies that:
- Universal AI access policies (broadband, cloud computing subsidies) are necessary but insufficient for cognitive augmentation clusters
- Institutional complementarity investments — data governance, regulatory clarity, educational alignment — are the binding constraints for regions in the low-augmentation equilibrium trap

### AI Governance

The Justice result (Paper 4) implies that:
- A welfare-maximizing AI governance framework includes a universal minimum augmentation access provision
- Augmentation inequality should be added to standard inequality monitoring frameworks
- AI governance institutions should be evaluated by their effect on the distribution of $C_i^A$ across the population, not just on GDP or employment effects

### Development and International Policy

The Trade result (Paper 5) implies that:
- Countries whose growth strategy depends on cognitive service exports (BPO, legal processing, financial analysis) face a structural risk if they have high $H^C$ relative to $H^A$
- Industrial policy for the AI era should prioritize $H^A$ development over general higher education expansion
- International AI governance agreements should address the distributional consequences of cross-country differences in augmentation capacity

---

## 12. Scope Conditions and Limitations

CFE is explicit about what it does and does not claim:

**Scope conditions for the AHC model:**
- The model applies to the regime of *generative* AI (post-2022) that can perform open-ended cognitive tasks. It does not apply to narrow AI (computer vision, recommendation systems) where the distinction between $H^C$ and $H^A$ operates differently.
- The augmentation function $\phi(D)$ is modeled as time-invariant. In practice, $\phi$ likely shifts as AI capabilities advance — this is a known simplification that future work should address.

**Scope conditions for the empirical work:**
- The LLM-based measurement pipeline uses English-language task descriptions from O\*NET as its primary training context. Application to occupational systems in other languages (ESCO, SISCO) requires careful translation validation.
- The household survey data used for estimation covers *formal* labor markets. The informal sector — a majority of workers in many Latin American countries — is partially observed and the transmission mechanisms may differ.

**Known limitations:**
- The distinction between $H^A$ and $H^C$ is theoretically clean but empirically fuzzy: many tasks have both substitutable and augmentable components. The LLM annotation assigns probabilities, not binary classifications, but the aggregation to occupation-level indices loses within-task heterogeneity.
- The augmentation function $\phi$ is estimated with firm-level data, but the mechanism operates at the task level. The mapping from task-level augmentation to firm-level productivity requires additional assumptions.

---

## 13. Glossary of CFE Concepts

| Term | Symbol | Definition |
|------|--------|------------|
| Augmentable Human Capital | $H_i^A$ | The component of cognitive capital that AI systems amplify rather than replace; characterized by tasks requiring judgment, contextual reasoning, novel synthesis |
| Routine Cognitive Capital | $H_i^C$ | The component of cognitive capital that AI systems substitute for; characterized by structured, repeatable information tasks |
| Physical-Manual Capital | $H_i^P$ | Embodied skills requiring physical presence; largely unaffected by generative AI |
| Digital Labor | $D_f$ | The stock of AI cognitive systems deployed by firm $f$; the AI counterpart of physical capital |
| Augmentation Function | $\phi(D)$ | The productivity multiplier that digital labor provides to $H^A$ workers; $\phi(0)=1$, $\phi$ increasing and concave |
| Augmentation Threshold | $D^*$ | The level of digital labor above which AI adoption increases wages for $H^A$-intensive workers on average |
| AHC Index | $AHC_o$ | LLM-generated measure of the augmentation potential of occupation $o$; the empirical proxy for $H_o^A$ |
| Cognitive Skill Half-Life | $t_{1/2}$ | Time for the economic value of a cognitive skill to depreciate to 50% of its initial value under the endogenous depreciation model |
| Augmentation Inequality | $\mathcal{I}^A$ | The distribution of the capability of augmentation $C_i^A$ across the population; analogous to income inequality |
| Capability of Augmentation | $C_i^A$ | Sen-inspired capability dimension: the effective ability of individual $i$ to amplify cognitive production via AI |
| Institutional Complementarity Distance | $\tau_{ij}$ | The gap between two regions' institutional profiles relevant for AI augmentation; replaces physical transport costs in the NEG model |
| Cognitive Comparative Advantage | $CCA_c$ | Country $c$'s relative productivity advantage in AI-augmented cognitive services, determined by $H^A/H^C$ composition |
| AI Washing Gap | $AWG_{ot}$ | The difference between perceived and actual AI exposure of occupation $o$ at time $t$; measures the attribution gap in corporate AI narratives |
| Augmentation Rent | $\rho^A$ | The excess return to $H^A$ capital above what would be predicted by standard Mincerian returns; generated by the scarcity of augmentable skills |

---

## 14. Citation Roadmap

The following are the primary references that CFE builds upon, situates itself against, or seeks to displace as the standard framework in its domain:

### Foundational References (CFE Extends)
- Becker, G. (1964). *Human Capital*. University of Chicago Press.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Krugman, P. (1991). "Increasing Returns and Economic Geography." *JPE*, 99(3).
- Sen, A. (1999). *Development as Freedom*. Oxford University Press.
- Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.

### Task-Based Literature (CFE Engages)
- Autor, D., Levy, F. & Murnane, R. (2003). "The Skill Content of Recent Technological Change." *QJE*, 118(4).
- Acemoglu, D. & Restrepo, P. (2018). "The Race between Man and Machine." *AER*, 108(6).
- Acemoglu, D. & Restrepo, P. (2022). "Tasks, Automation, and the Rise in US Wage Inequality." *Econometrica*, 90(5).

### AI Measurement Literature (CFE Supersedes)
- Frey, C.B. & Osborne, M. (2017). "The Future of Employment." *Technological Forecasting and Social Change*.
- Felten, E., Raj, M. & Seamans, R. (2021). "Occupational, Industry, and Geographic Exposure to AI." AEI Working Paper.
- Eloundou, T. et al. (2023). "GPTs are GPTs." NBER Working Paper.
- Massenkoff, M. & McCrory, P. (2026). "Labor Market Impacts of AI: A New Measure and Early Evidence." Anthropic Research.

### AI and Growth (CFE Engages)
- Bresnahan, T. & Trajtenberg, M. (1995). "General Purpose Technologies." *Journal of Econometrics*.
- Acemoglu, D. (2024). "The Simple Macroeconomics of AI." NBER Working Paper.
- Korinek, A. & Suh, D. (2024). "Scenarios for the Transition to AGI." *Brookings*.
- Groweic, J. (2024). "A New Perspective on the Past and Future of Economic Growth." CEPR.

### Development and Trade Context
- Cerutti, E. et al. (2025). "TFP Impacts of AI across Sectors." IMF Working Paper.
- Cazzaniga, M. et al. (2024). "Gen-AI: Artificial Intelligence and the Future of Work." IMF SDN.
- WTO (2025). "AI and Trade." Staff Working Paper ERSD-2025-09.

---

## 15. Author and Affiliation

**Cristian Espinal** ([@Cespial](https://github.com/Cespial))
- Founder & Director, INPLUX SAS — Medellín, Colombia
- Jefe de Arquitectura Tecnológica e IA, Redek
- Professor, Postgraduate Division — ESUMER
- Research Associate, CIIID (Minciencias-recognized)
- M.A. in Economics — Universidad EAFIT
- Doctoral candidate (Engineering, in progress)

**Research context:** CFE emerges from the intersection of Cristian's applied AI/ML work (TribAI, GovTech systems, RAG pipelines) with his academic background in economics and his experience at the frontier of AI deployment in developing-economy institutional contexts. The program is grounded in the observation that standard economic frameworks systematically mischaracterize what happens to human labor when AI systems become productive collaborators rather than mere automation tools — an observation made vivid by building such systems in practice.

**Affiliated projects:**
- TribAI (tribai.co) — Colombian tax intelligence platform; primary testbed for LLM measurement methodology
- INPLUX GovTech Suite — AI systems for Colombian public sector; source of institutional complementarity insights
- Tensor Analytics — Territorial intelligence; spatial econometrics for the geography paper

---

## 16. Current Status — Paper 1 Pipeline (March 2026)

Paper 1 (Augmented Human Capital) is the first paper in active development. The empirical pipeline is operational:

### Data Acquired
| Dataset | Records | Status |
|---------|---------|--------|
| O*NET 30.2 (41 files) | 19K task statements, 52 abilities, 900+ occupations | Downloaded |
| GEIH 2024 (DANE) | 111,672 workers with CIUO-08 4-digit codes | Loaded from upstream |
| EAM 2016-2024 (DANE) | 59,908 firm-year observations | Loaded from upstream |
| Penn World Table 11.0 | 185 countries, 1950-2023 | Available |
| Felten AIOE, Webb AI exposure | 774 SOC occupations | Pending (URL fix) |

### Pipeline Status
| Stage | Output | Rows | Status |
|-------|--------|------|--------|
| SOC → ISCO crosswalk | `soc_isco.parquet` | 1,016 | Done |
| ISCO → CIUO crosswalk | `isco_ciuo.parquet` | 443 | Done |
| Chained crosswalk | `soc_ciuo_chained.parquet` | 66,157 | Done |
| Task-occupation matrix | `occupation_task_matrix.parquet` | 2,229,863 | Done |
| H^P/H^C/H^A classification | `task_classification.parquet` | 2,229,863 | Done (rules-based) |
| **LLM scoring (Haiku)** | `raw_llm_scores.jsonl` | **18,796 tasks** | **In progress** |
| AHC index aggregation | `ahc_index_by_occupation.parquet` | — | Pending (after scoring) |
| Estimation sample | `estimation_sample.parquet` | 105,517 | Done (AHC placeholder) |

### Key Early Results
- Task classification distribution: **H^A 71%, H^P 21%, H^C 8%**
- LLM scoring (first 160 tasks): Augmentation mean=62/100, Substitution mean=38/100
- Dominant augmentation types: Decision Support (51%), Information Synthesis (42%)
- Zero scoring errors so far

### Next Steps
1. Complete LLM scoring (~18K tasks remaining, ~6 hours)
2. Aggregate AHC index by occupation
3. Validate against Felten AIOE and Frey-Osborne
4. Estimate augmented Mincer equation (OLS + IV)
5. Write Sections 1-2 (Theory) in parallel

---

## License and Use

This research agenda is the intellectual property of Cristian Espinal / INPLUX SAS. Working papers, datasets, and code produced under the CFE program will be released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) unless otherwise noted.

Academic citations should reference: *Espinal, C. (2026). "Cognitive Factor Economics: A Research Program on the Economics of Human-AI Complementarity." INPLUX Working Paper Series.*

---

*Cognitive Factor Economics — INPLUX Research Division*
*Medellín, Colombia · 2026*
*Last updated: March 31, 2026*

---
