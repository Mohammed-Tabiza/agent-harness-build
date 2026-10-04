# Construire un agent, étape par étape

Ce projet propose six scripts Python pour comprendre comment construire un assistant capable de consulter des fichiers, d’agir avec des outils et de conserver des informations. Chaque version isole une notion ; `harness.py` rassemble les principaux mécanismes dans une interface en ligne de commande.

Un **harness** est le programme qui entoure le modèle : il prépare les messages, décrit les outils disponibles, exécute les appels demandés et renvoie leurs résultats au modèle.

## Progression pédagogique

Ordre de lecture conseillé : **`v0.py` → `v1.py` → `v2.py` → `v3.py` → `v4.py` → `harness.py`**.

| Script | Ce qu’il fait | Notion à apprendre |
| --- | --- | --- |
| [`v0.py`](v0.py) | Envoie une question au modèle et affiche sa réponse. | Un appel au modèle et les rôles des messages. |
| [`v1.py`](v1.py) | Expose des outils pour lister et lire les fichiers, puis effectue un seul cycle d’exécution d’outils. | Le fonctionnement du *tool calling*. |
| [`v2.py`](v2.py) | Ajoute l’écriture de fichiers et répète les appels au modèle jusqu’à une réponse finale. | La boucle agentique et les actions en plusieurs étapes. |
| [`v3.py`](v3.py) | Connecte des serveurs MCP et combine leurs outils avec les outils locaux. | La découverte d’outils externes et les sessions asynchrones. |
| [`v4.py`](v4.py) | Conserve l’historique de la conversation et mémorise des faits dans `memory.md`. | La différence entre contexte de session et mémoire persistante. |
| [`harness.py`](harness.py) | Réunit outils locaux, MCP, mémoire et changement de modèle pendant la conversation. | L’intégration des mécanismes dans un assistant configurable. |

Les versions ne sont pas strictement cumulatives : `v4.py` revient à des outils locaux pour se concentrer sur la mémoire. Il ne reprend ni MCP ni l’outil d’écriture de fichiers de `v2.py`. `harness.py` réunit ensuite ces capacités.

## Rôle de chaque script

### `v0.py` — Dialoguer avec un modèle

La fonction `chat()` utilise le client `OpenAI` avec un endpoint Ollama compatible avec l’API OpenAI, à l’adresse `http://localhost:11434/v1`. Elle transmet un message système et la question de l’utilisateur, puis renvoie le texte de la réponse.

La boucle du terminal permet de poser plusieurs questions, mais chaque appel recrée les messages : le modèle ne reçoit pas les échanges précédents. Aucun outil n’est disponible.

**Exercice :** poser une question, puis demander « Que viens-je de te demander ? » pour observer l’absence d’historique.

### `v1.py` — Donner accès à des outils

Deux fonctions Python deviennent des outils :

- `list_files()` liste les éléments de `workspace/`.
- `read_file(filename)` lit un fichier de ce dossier.

`TOOL_SCHEMAS` décrit ces fonctions et leurs paramètres au modèle ; `TOOLS` associe leur nom à leur implémentation Python. Le modèle propose des appels, le script les exécute, puis transmet les résultats avec le rôle `tool` et le `tool_call_id` correspondant.

Cette version traite un seul lot d’appels d’outils. Le second appel au modèle demande une réponse finale sans lui proposer à nouveau les outils. Elle ne permet donc pas un enchaînement où le modèle liste d’abord les fichiers, puis décide de lire un nom découvert dans le résultat.

**Exercice :** créer `workspace/notes.txt`, puis demander « Lis notes.txt et résume son contenu ». Repérer l’appel d’outil affiché dans le terminal.

### `v2.py` — Répéter jusqu’à terminer la tâche

Cette version ajoute `write_file(filename, content)`, qui crée ou remplace un fichier. La fonction `run_agent()` introduit une boucle :

1. Envoyer les messages et les descriptions d’outils au modèle.
2. Exécuter les outils demandés et ajouter leurs résultats aux messages.
3. Recommencer tant que le modèle demande des outils.
4. Renvoyer la réponse lorsque le modèle ne demande plus d’outil.

L’agent peut ainsi découvrir des fichiers, en lire certains, puis écrire un résultat. Les résultats intermédiaires restent disponibles pendant la tâche, mais l’historique est recréé à chaque nouvelle question du terminal.

**Exercice :** demander « Liste les fichiers, lis notes.txt et écris un résumé dans resume.txt ». Observer plusieurs cycles et inspecter le fichier produit.

### `v3.py` — Brancher des outils externes avec MCP

MCP (*Model Context Protocol*) permet de découvrir et d’appeler les outils de processus externes. Ce script configure deux serveurs, lancés avec `uvx` : `mcp-server-time` et `mcp-server-fetch`.

`connect_mcp()` démarre les serveurs via leur entrée/sortie standard, initialise les sessions et récupère leurs descriptions d’outils. Ces descriptions rejoignent `TOOL_SCHEMAS`. `call_tool()` choisit ensuite entre une fonction locale et un appel à une session MCP.

Le script utilise `AsyncOpenAI`, `async`/`await` et `AsyncExitStack` pour gérer les connexions. L’entrée du terminal passe par `asyncio.to_thread()`. La boucle agentique reste la même dans son principe ; chaque question conserve son propre contexte.

**Exercice :** demander l’heure dans un fuseau donné, puis le contenu d’une page publique. Observer les outils découverts au démarrage et ceux appelés pendant la tâche.

### `v4.py` — Conserver une mémoire

Cette version distingue deux formes de mémoire :

- **Historique de session :** une liste `messages` partagée entre les tours conserve les questions, réponses et échanges avec les outils pendant l’exécution.
- **Mémoire persistante :** `save_memory(fact)` ajoute un fait dans `memory.md`, et `load_memory()` charge ce fichier dans le message système au démarrage suivant.

Les outils disponibles sont `list_files`, `read_file` et `save_memory`. Le message système invite le modèle à enregistrer les faits durables sur l’utilisateur. La mémoire n’est ni une base vectorielle ni un mécanisme de recherche : c’est un fichier Markdown injecté dans le contexte initial.

**Exercice :** demander « Mémorise que je préfère les réponses en français », vérifier `memory.md`, puis relancer le script et interroger l’assistant sur cette préférence. L’enregistrement dépend de l’appel effectif à `save_memory`.

### `harness.py` — Assembler un assistant configurable

Le script final réunit :

- les outils locaux de lecture, de liste et d’écriture, ainsi que `save_memory` ;
- la mémoire persistante et l’historique de conversation ;
- les serveurs MCP déclarés dans `mcp_servers.json` ;
- un registre `MODELS`, dont chaque configuration définit un endpoint, une clé et un nom de modèle ;
- une boucle agentique asynchrone et des commandes interactives.

Le modèle initial est `local`. Les entrées `local` et `local-small` ciblent Ollama ; `gpt` et `claude` sont les configurations distantes présentes dans le code. Changer de modèle conserve la même liste de messages et les mêmes outils. La disponibilité effective des modèles et la compatibilité des endpoints doivent être vérifiées dans l’environnement utilisé.

| Commande | Effet |
| --- | --- |
| `/models` | Affiche les configurations et indique le modèle actif. |
| `/model <nom>` | Sélectionne une entrée du registre, par exemple `/model local-small`. |
| `/tools` | Affiche les outils et leur origine locale ou MCP. |
| `/memory` | Affiche le contenu actuel de `memory.md`. |
| `/quit` | Termine le programme. |

**Exercice :** consulter `/tools`, demander un résumé de fichier, changer de modèle avec `/model`, puis poursuivre la conversation pour comparer les réponses avec le même contexte.

## Préparer et lancer les exemples

Prévoir Python, un environnement virtuel, un endpoint Ollama accessible pour les configurations locales et un modèle correspondant à la constante `MODEL` ou au registre `MODELS`. Les scripts `v0.py` à `v4.py` utilisent actuellement `gemma4:31b-cloud` ; adapter cette constante si nécessaire. Le modèle choisi doit prendre en charge les appels d’outils pour les versions qui les utilisent.

Depuis la racine du projet, sous PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install openai mcp pydantic
New-Item -ItemType Directory -Force workspace
Set-Content -Path workspace/notes.txt -Value 'Apprendre les outils, puis la boucle agentique.' -Encoding utf8
python v0.py
```

Lancer ensuite le script souhaité, par exemple `python v2.py`, `python v4.py` ou `python harness.py`. Les fichiers sont recherchés dans le dossier `workspace/` situé à côté des scripts. `memory.md` est également situé à côté des scripts et est créé lors du premier enregistrement d’un fait.

Pour `v3.py`, `uvx` doit être disponible afin de lancer les deux serveurs configurés. Pour `harness.py`, créer ou vérifier le fichier `mcp_servers.json` à la racine. Pour commencer sans serveur MCP, son contenu minimal est :

```json
{
  "mcpServers": {}
}
```

Pour reprendre les serveurs de `v3.py` :

```json
{
  "mcpServers": {
    "time": { "command": "uvx", "args": ["mcp-server-time"] },
    "fetch": { "command": "uvx", "args": ["mcp-server-fetch"] }
  }
}
```

Les configurations distantes de `harness.py` lisent respectivement `OPENAI_API_KEY` et `ANTHROPIC_API_KEY` dans l’environnement. Elles ne sont nécessaires que pour utiliser les entrées correspondantes.

## Ce qu’il faut retenir

Le modèle choisit les appels d’outils ; le programme les exécute. La boucle permet au modèle de réagir à leurs résultats. MCP étend le catalogue d’outils, la mémoire conserve des informations entre les sessions et le registre de modèles permet de changer le moteur sans réécrire cette orchestration.

Ces scripts privilégient la lisibilité pédagogique. `write_file` remplace le contenu existant ; les fonctions de fichiers assemblent directement les chemins sans contrôle de confinement ; les boucles ne fixent pas de nombre maximal d’itérations. L’historique complet n’est pas enregistré sur disque et les faits de `memory.md` sont chargés dans le message système au démarrage, sans rechargement automatique pendant la session.
