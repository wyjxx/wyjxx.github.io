# **GeoMindMap: Investigating LLM Reasoning in Geo-localization**

GeoMindMap is an end-to-end pipeline and visualization framework designed to make the reasoning process of Large Language Models (LLMs) interpretable in the context of image-based geo-localization tasks.

This project was developed as part of my Bachelor’s Thesis at TUM (“Tracing Locations Through Words and Images: Investigating Large Language Model Reasoning in Geo-localization”).

## **Motivation**

LLMs demonstrate remarkable reasoning ability, but their reasoning process is often a black box—especially in multi-modal tasks such as geo-localization.

Geo-localization (e.g., “Where was this photo taken?”) is a challenging problem because:

It requires multi-step reasoning: observe → hypothesize → verify → conclude.

It involves heterogeneous evidence: visual clues, inferred knowledge, and candidate locations.

It reveals human-like reasoning features such as breadth-first exploration, depth-first verification, and strategy switching.

GeoMindMap addresses this challenge by:

Decomposing raw reasoning text into structured entities

Visualizing reasoning dynamically as two evolving maps (Clue Map + Location Map)

Identifying reasoning features that correlate with accuracy and efficiency

## **Pipeline Overview**

The GeoMindMap pipeline consists of five main modules:

### 1. Reasoning Segmentation

Split LLM’s reasoning output into step-wise paragraphs.

Enables dynamic analysis.

### 2. Entity Extraction

Extract three types of entities from reasoning:

v: visual elements (e.g., “red banner”, “mountain range”)

i: inferential knowledge (e.g., “Baroque style”, “suburban area”)

l: locations (e.g., “Germany”, “Munich”, “Marienplatz”)

### 3. Granularity & Parent Assignment

Assign each entity a granularity level:

Visual/Inference: coarse (scene) → fine (detail)

Locations: continent → country → region → city → street

Build hierarchical structures for visualization.

### 4. Step-wise Semantic Matching

Match entities to reasoning paragraphs.

Update candidate locations with status: excluded, included, concluded.

Link supporting clues to each decision.

### 5. Visualization

Clue Map: shows visual and inferential entities as concentric layers.

Location Map: shows geographic entities hierarchically.

Interactive D3.js web interface supports step-by-step playback.

## **Unique Contributions**

### Entity-Level Reasoning Visualization
Not just paragraphs → fine-grained decomposition into entities, hierarchies, and links.

### Dual Maps

Clue Map = “decision support side”

Location Map = “decision making side”

Together, they reveal how LLMs connect observation to conclusion.

## **Reasoning Feature Analysis**
Identifies 5 reasoning features:

Self-Reflection

Reasoning Gap

Breadth-First (BF)

Depth-First (DF)

Strategy Switch

## **Cross-Model Benchmarking**
Evaluates OpenAI o4-mini, Gemini 2.5 Pro, Claude Sonnet 4 on dataset of 86 smartphone photos.

All achieve ~city-level accuracy

But show distinct efficiency–performance trade-offs and reasoning strategies

## **Example**

### Input:
Smartphone photo of Würzburg, Germany

### Reasoning (raw LLM output):

“This looks like a European old town… Ratskeller sign → Germany… Could be Regensburg or Würzburg… The fountain matches Vierroehrenbrunnen → Final conclusion: Würzburg.”

### Visualization (GeoMindMap):

Clue Map: Ratskeller sign, Baroque style, town hall, fountain

Location Map: Europe → Germany → Bavaria → Würzburg

Step-wise updates: nodes highlighted, candidate locations excluded/confirmed

## **Repository Structure**
<pre> ```
geomindmap/
│
├── data/
│ └── JSON outputs of different models (entities, vi_map, l_map, para_match, etc.)
│
├── pictures/
│ └── 86 filitered images used for geo-localization tasks
│
├── pipeline/
│ ├── reasoning.py # Generate step-wise reasoning traces from LLMs
│ ├── extract.py # Extract entities and build hierarchical maps
│ ├── match.py # Match reasoning steps with entities
│ ├── coordinate.py # Compute 2D coordinates for visualization
│ └── main.py # Orchestrate the full pipeline
│
├── index.html # Interactive webpage visualization
└── README.md # Project documentation
``` </pre>

## **Acknowledgements**

This project was developed as part of my Bachelor’s Thesis at TUM (Tracing Locations Through Words and Images), supervised by Dr. Mark Huasong Meng and Prof. Dr. Chunyang Chen.
