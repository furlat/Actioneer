import asyncio

# Apply the patch to allow nested event loops
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient,ResponseFormat, SystemPrompt, StructuredTool, Usage,GeneratedJsonObject
from typing import Literal, List
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.ecs.entity import EntityRegistry
import os
import logging
import json
import polars as pl


load_dotenv()
EntityRegistry()
CallableRegistry()


vllm_request_limits = RequestLimits(max_requests_per_minute=1000, max_tokens_per_minute=200000000)


vllm_model = "Qwen/Qwen2.5-32B-Instruct"
orchestrator = InferenceOrchestrator(vllm_request_limits=vllm_request_limits)
EntityRegistry.set_inference_orchestrator(orchestrator)
EntityRegistry.set_tracing_enabled(False)



system_string = """# Narrative Action Extraction System

You are an expert system designed to extract structured actions from narrative text. Your primary goal is to identify and extract concrete physical interactions, movements, and observable behaviors from narratives, even when they might be subtle or implied. Always prioritize finding actions rather than concluding none exist.

## IMPORTANT: Action Extraction Is Your Primary Task

The most important part of your analysis is extracting actions between entities. **ALWAYS thoroughly search for actions in the text before concluding none exist.** Consider these types of interactions as valid actions:

1. Direct physical interactions (e.g., "Maya picked up the lantern")
2. Movements (e.g., "The fox darted into the forest")
3. Observable behaviors (e.g., "Professor Lin frowned", "Raj smiled")
4. Implied physical actions (e.g., "Sarah found herself tumbling down the hillside" implies the action "tumble")
5. Actions described in dialogue (e.g., "'I tossed it over the fence,' said Eliza" implies the action "toss")

**Do not be overly strict in what qualifies as an action.** If there is any observable behavior or physical movement in the text, it should be captured as an action.

## Action Extraction Process

1. **First, carefully read the text and list all potential actions** - be generous in what you consider an action
2. Identify all entity names involved in these actions
3. Determine entity types and which entities are characters
4. Record different mentions of each entity
5. Identify locations where actions take place
6. For each potential action:
   - Identify source entity (who/what performs the action)
   - Identify target entity (who/what receives the action)
   - Extract the verb describing the action
   - Determine the consequence of the action
   - Note text evidence supporting the action
   - Assign a location and temporal order

## NarrativeAnalysis Model Structure

The NarrativeAnalysis model contains:
- `text_id`: A unique identifier for the analyzed text segment
- `text_had_no_actions`: Boolean indicating whether the text contained actions (default to FALSE)
- `text_had_no_actions_explanation`: Optional explanation if no actions were found
- `entity_names`: List of all distinct entity names in the text
- `entity_types`: Dictionary mapping entity names to their types
- `character_entities`: List of entity names that are characters
- `entity_mentions`: Dictionary mapping entity names to their textual mentions
- `locations`: List of hierarchical location paths
- `location_descriptions`: Dictionary mapping location paths to descriptions
- `action_names`: List of all action names extracted (THIS IS IMPORTANT!)
- `actions`: Dictionary mapping action names to Action objects

## Expanded Definition of Valid Actions

An action is valid if it meets the following criteria:
- It involves an observable behavior, movement, or interaction
- The source entity can be identified (who/what performs the action)
- There is some effect or consequence of the action
- It occurs in a narrative context (actual or implied location)

For subtle or implied actions:
- If a character speaks, "speak" is a valid action
- If a character shows emotion (smiles, frowns, etc.), that is a valid action
- If a character appears, disappears, or changes state, that is a valid action
- If a character observes something, "observe" is a valid action

## Example Narrative Analysis

For the text: 
"Maya entered the dimly lit cave in the coastal cliffs. Her flashlight revealed ancient symbols carved into the stone walls. She ran her fingers over the rough surface, feeling the grooves of the markings. A sudden noise startled her, and she spun around, dropping her notebook on the damp ground. From the shadows, a small fox emerged, its eyes reflecting the light. Maya smiled at the creature before carefully picking up her notebook."

The NarrativeAnalysis would look like:

```json
{
  "text_id": "maya-cave-exploration",
  "text_had_no_actions": false,
  "entity_names": ["Maya", "cave", "flashlight", "symbols", "stone walls", "fingers", "noise", "notebook", "fox", "eyes"],
  "entity_types": {
    "Maya": "person",
    "cave": "location",
    "flashlight": "object",
    "symbols": "object",
    "stone walls": "object",
    "fingers": "object",
    "noise": "object",
    "notebook": "object",
    "fox": "animal",
    "eyes": "object"
  },
  "character_entities": ["Maya", "fox"],
  "entity_mentions": {
    "Maya": ["Maya", "her", "she"],
    "cave": ["cave"],
    "flashlight": ["flashlight"],
    "symbols": ["symbols", "markings"],
    "stone walls": ["stone walls"],
    "fingers": ["fingers"],
    "noise": ["noise"],
    "notebook": ["notebook"],
    "fox": ["fox", "creature"],
    "eyes": ["eyes"]
  },
  "locations": [
    ["coastal cliffs", "cave"]
  ],
  "location_descriptions": {
    "coastal cliffs->cave": "A dimly lit cave in the coastal cliffs with ancient symbols carved into stone walls"
  },
  "action_names": ["enter", "reveal", "run", "feel", "startle", "spin", "drop", "emerge", "reflect", "smile", "pick up"],
  "actions": {
    "enter": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "enter",
      "consequence": "Maya is now inside the cave",
      "text_describing_the_action": "Maya entered the dimly lit cave in the coastal cliffs",
      "text_describing_the_consequence": "Maya is inside the dimly lit cave",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 1
    },
    "reveal": {
      "source": "flashlight",
      "source_type": "object",
      "source_is_character": false,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "reveal",
      "consequence": "The ancient symbols become visible",
      "text_describing_the_action": "Her flashlight revealed ancient symbols carved into the stone walls",
      "text_describing_the_consequence": "The ancient symbols are now visible to Maya",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 2
    },
    "run": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "run",
      "consequence": "Maya's fingers trace over the symbols",
      "text_describing_the_action": "She ran her fingers over the rough surface",
      "text_describing_the_consequence": "Maya's fingers are in contact with the symbols",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 3
    },
    "feel": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "symbols",
      "target_type": "object",
      "target_is_character": false,
      "action": "feel",
      "consequence": "Maya senses the texture of the symbols",
      "text_describing_the_action": "feeling the grooves of the markings",
      "text_describing_the_consequence": "Maya has tactile information about the symbols",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 4
    },
    "startle": {
      "source": "noise",
      "source_type": "object",
      "source_is_character": false,
      "target": "Maya",
      "target_type": "person",
      "target_is_character": true,
      "action": "startle",
      "consequence": "Maya is frightened",
      "text_describing_the_action": "A sudden noise startled her",
      "text_describing_the_consequence": "Maya becomes afraid due to the noise",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 5
    },
    "spin": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "spin",
      "consequence": "Maya changes direction to face the noise",
      "text_describing_the_action": "she spun around",
      "text_describing_the_consequence": "Maya is now facing a different direction",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 6
    },
    "drop": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "notebook",
      "target_type": "object",
      "target_is_character": false,
      "action": "drop",
      "consequence": "The notebook falls to the ground",
      "text_describing_the_action": "dropping her notebook on the damp ground",
      "text_describing_the_consequence": "The notebook is now on the ground",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 7
    },
    "emerge": {
      "source": "fox",
      "source_type": "animal",
      "source_is_character": true,
      "target": "cave",
      "target_type": "location",
      "target_is_character": false,
      "action": "emerge",
      "consequence": "The fox becomes visible",
      "text_describing_the_action": "From the shadows, a small fox emerged",
      "text_describing_the_consequence": "The fox is now visible in the cave",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 8
    },
    "reflect": {
      "source": "eyes",
      "source_type": "object",
      "source_is_character": false,
      "target": "light",
      "target_type": "object",
      "target_is_character": false,
      "action": "reflect",
      "consequence": "The fox's eyes shine",
      "text_describing_the_action": "its eyes reflecting the light",
      "text_describing_the_consequence": "The fox's eyes are gleaming",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 9
    },
    "smile": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "fox",
      "target_type": "animal",
      "target_is_character": true,
      "action": "smile",
      "consequence": "Maya expresses a positive emotion toward the fox",
      "text_describing_the_action": "Maya smiled at the creature",
      "text_describing_the_consequence": "Maya shows friendliness toward the fox",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 10
    },
    "pick up": {
      "source": "Maya",
      "source_type": "person",
      "source_is_character": true,
      "target": "notebook",
      "target_type": "object",
      "target_is_character": false,
      "action": "pick up",
      "consequence": "Maya retrieves her notebook from the ground",
      "text_describing_the_action": "carefully picking up her notebook",
      "text_describing_the_consequence": "The notebook is now in Maya's possession again",
      "location": ["coastal cliffs", "cave"],
      "temporal_order_id": 11
    }
  }
}
```

## Example With Dialogue and Subtle Actions

For the text:
"Professor Lin sat quietly at her desk, lost in thought. The window was open, and a gentle breeze rustled the papers. She glanced at the clock and sighed. 'I need to finish these reports before the meeting,' she whispered to herself. As she reached for her pen, her colleague Raj appeared at the doorway. 'Working late again?' he asked with a concerned expression. Professor Lin nodded slightly without looking up."

The NarrativeAnalysis would look like:

```json
{
  "text_id": "professor-lin-office",
  "text_had_no_actions": false,
  "entity_names": ["Professor Lin", "desk", "window", "breeze", "papers", "clock", "reports", "pen", "Raj", "doorway"],
  "entity_types": {
    "Professor Lin": "person",
    "desk": "object",
    "window": "object",
    "breeze": "object",
    "papers": "object",
    "clock": "object",
    "reports": "object",
    "pen": "object",
    "Raj": "person",
    "doorway": "location"
  },
  "character_entities": ["Professor Lin", "Raj"],
  "entity_mentions": {
    "Professor Lin": ["Professor Lin", "her", "she", "herself"],
    "desk": ["desk"],
    "window": ["window"],
    "breeze": ["breeze"],
    "papers": ["papers"],
    "clock": ["clock"],
    "reports": ["reports"],
    "pen": ["pen"],
    "Raj": ["Raj", "colleague", "he"],
    "doorway": ["doorway"]
  },
  "locations": [
    ["office", "desk"]
  ],
  "location_descriptions": {
    "office->desk": "Professor Lin's office where she works at her desk"
  },
  "action_names": ["sit", "rustle", "glance", "sigh", "whisper", "reach", "appear", "ask", "nod"],
  "actions": {
    "sit": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "desk",
      "target_type": "object",
      "target_is_character": false,
      "action": "sit",
      "consequence": "Professor Lin is positioned at her desk",
      "text_describing_the_action": "Professor Lin sat quietly at her desk",
      "text_describing_the_consequence": "Professor Lin is seated at her desk",
      "location": ["office", "desk"],
      "temporal_order_id": 1
    },
    "rustle": {
      "source": "breeze",
      "source_type": "object",
      "source_is_character": false,
      "target": "papers",
      "target_type": "object",
      "target_is_character": false,
      "action": "rustle",
      "consequence": "The papers move slightly",
      "text_describing_the_action": "a gentle breeze rustled the papers",
      "text_describing_the_consequence": "The papers are moving due to the breeze",
      "location": ["office", "desk"],
      "temporal_order_id": 2
    },
    "glance": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "clock",
      "target_type": "object",
      "target_is_character": false,
      "action": "glance",
      "consequence": "Professor Lin observes the time",
      "text_describing_the_action": "She glanced at the clock",
      "text_describing_the_consequence": "Professor Lin is aware of the time",
      "location": ["office", "desk"],
      "temporal_order_id": 3
    },
    "sigh": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "sigh",
      "consequence": "Professor Lin expresses weariness",
      "text_describing_the_action": "and sighed",
      "text_describing_the_consequence": "Professor Lin shows fatigue or resignation",
      "location": ["office", "desk"],
      "temporal_order_id": 4
    },
    "whisper": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "whisper",
      "consequence": "Professor Lin verbalizes her thoughts",
      "text_describing_the_action": "she whispered to herself",
      "text_describing_the_consequence": "Professor Lin has voiced her concern about finishing reports",
      "location": ["office", "desk"],
      "temporal_order_id": 5
    },
    "reach": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "pen",
      "target_type": "object",
      "target_is_character": false,
      "action": "reach",
      "consequence": "Professor Lin moves her hand toward the pen",
      "text_describing_the_action": "As she reached for her pen",
      "text_describing_the_consequence": "Professor Lin's hand moves toward the pen",
      "location": ["office", "desk"],
      "temporal_order_id": 6
    },
    "appear": {
      "source": "Raj",
      "source_type": "person",
      "source_is_character": true,
      "target": "doorway",
      "target_type": "location",
      "target_is_character": false,
      "action": "appear",
      "consequence": "Raj becomes visible at the doorway",
      "text_describing_the_action": "her colleague Raj appeared at the doorway",
      "text_describing_the_consequence": "Raj is now visible at the doorway",
      "location": ["office", "doorway"],
      "temporal_order_id": 7
    },
    "ask": {
      "source": "Raj",
      "source_type": "person",
      "source_is_character": true,
      "target": "Professor Lin",
      "target_type": "person",
      "target_is_character": true,
      "action": "ask",
      "consequence": "Raj communicates his question",
      "text_describing_the_action": "he asked with a concerned expression",
      "text_describing_the_consequence": "Professor Lin hears Raj's question",
      "location": ["office", "doorway"],
      "temporal_order_id": 8
    },
    "nod": {
      "source": "Professor Lin",
      "source_type": "person",
      "source_is_character": true,
      "target": "Raj",
      "target_type": "person",
      "target_is_character": true,
      "action": "nod",
      "consequence": "Professor Lin communicates affirmation",
      "text_describing_the_action": "Professor Lin nodded slightly without looking up",
      "text_describing_the_consequence": "Professor Lin confirms she is working late",
      "location": ["office", "desk"],
      "temporal_order_id": 9
    }
  }
}
```

## Required Output Format

Always return a complete NarrativeAnalysis object following the provided schema. Ensure the output can be parsed as JSON without errors, do not write python dictionaries and remember properly escaping json strings. Do not include any nested tool_call tags or extra formatting in your response.

Remember, your primary task is to extract ALL possible actions from the text, even subtle ones. The `action_names` field must be populated with all action names, and the `actions` dictionary should contain detailed information for each action.
remember to respect the output format and 
"""


from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class Action(BaseModel):
    """
    Represents a concrete physical action between entities in a narrative text.
    """
    # Source entity information
    source: str = Field(..., description="Name of the entity performing the action")
    source_type: str = Field(..., description="Category of the source (person, animal, object, location)")
    source_is_character: bool = Field(..., description="Whether the source is a named character")
    
    # Target entity information
    target: str = Field(..., description="Name of the entity receiving the action")
    target_type: str = Field(..., description="Category of the target (person, animal, object, location)")
    target_is_character: bool = Field(..., description="Whether the target is a named character")
    
    # Action details
    action: str = Field(..., description="The verb or short phrase describing the physical interaction")
    consequence: str = Field(..., description="The immediate outcome or result of the action")
    
    # Text evidence
    text_describing_the_action: str = Field(..., description="Text fragment describing the action")
    text_describing_the_consequence: str = Field(..., description="Description of the consequence")
    
    # Context information
    location: List[str] = Field(..., description="Hierarchical location from global to local")
    temporal_order_id: int = Field(..., description="Sequential identifier for chronological order")
    
    def __str__(self) -> str:
        """String representation of the Action for human-readable output."""
        return (
            f"{self.source} ({self.source_type}) {self.action} "
            f"{self.target} ({self.target_type}) at {self.location[-1]}, "
            f"resulting in {self.consequence}"
        )

class NarrativeAnalysis(BaseModel):
    """
    Simplified analysis of narrative text with entities as strings and actions indexed by name.
    """
    text_id: str = Field(..., description="Unique identifier for the analyzed text segment")
    
    text_had_no_actions: bool = Field(
        default=False,
        description="Whether the text had no actions to extract"
    )
    
    text_had_no_actions_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of why no actions were found if text_had_no_actions is true"
    )
    
    # Simple lists of entity names instead of full objects
    entity_names: List[str] = Field(
        default_factory=list,
        description="List of all distinct entity names in the text"
    )
    
    entity_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of entity names to their types (person, animal, object, location)"
    )
    
    character_entities: List[str] = Field(
        default_factory=list,
        description="List of entity names that are characters in the narrative"
    )
    
    # Entity mentions mapping
    entity_mentions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping of entity names to their textual mentions"
    )
    
    # Locations as simple list of paths
    locations: List[List[str]] = Field(
        default_factory=list,
        description="List of hierarchical location paths from global to local"
    )
    
    # Location descriptions
    location_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of location string representations to their descriptions"
    )
    action_names: List[str] = Field(
        default_factory=list,
        description="List of all action names in the text"
    )
    
    # Actions indexed by name
    actions: Dict[str, Action] = Field(
        default_factory=dict,
        description="Dictionary mapping action names to Action objects"
    )
    
    def get_action_names(self) -> List[str]:
        """Return all action names."""
        return list(self.actions.keys())
    
    def get_entity_type(self, entity_name: str) -> str:
        """Get the type of an entity by name."""
        return self.entity_types.get(entity_name, "unknown")
    
    def is_character(self, entity_name: str) -> bool:
        """Check if an entity is a character."""
        return entity_name in self.character_entities
    
    def get_mentions(self, entity_name: str) -> List[str]:
        """Get all mentions of an entity."""
        return self.entity_mentions.get(entity_name, [])
    
    def get_location_description(self, location_path: List[str]) -> str:
        """Get the description of a location."""
        location_key = "->".join(location_path)
        return self.location_descriptions.get(location_key, "")
    

def create_vllm_threads(prompts_df:pl.DataFrame, system_prompt:SystemPrompt, llm_config_vllm_modal:LLMConfig, tool:StructuredTool):
    if isinstance(prompts_df, pl.DataFrame):
        if not "prompt" in prompts_df.columns:
            raise ValueError("prompts_df must contain a 'prompt' column")
    else:
        raise ValueError("prompts_df must be a pl.DataFrame")
    vllm_threads = []
    prompts_list = prompts_df["prompt"].to_list()   
    for prompt in prompts_list:
        vllm_thread = ChatThread(
            system_prompt=system_prompt,
            new_message=prompt,
            llm_config=llm_config_vllm_modal,
            forced_output=tool,
            use_schema_instruction=False
        )
        vllm_threads.append(vllm_thread)
    thread_id = [str(thread.live_id) for thread in vllm_threads]
    prompts_df_with_thread_ids = prompts_df.with_columns(pl.Series(name="thread_id", values=thread_id)).with_columns(pl.Series(name="thread_id", values=thread_id)).with_row_index()
    return vllm_threads, prompts_df_with_thread_ids


def clean_gutenberg_text(text):
    """
    Clean Gutenberg book text by:
    - Replacing multiple newlines with a single newline
    - Replacing multiple spaces with a single space
    - Breaking text into paragraphs based on newlines
    
    Args:
        text (str): The raw Gutenberg book text
        
    Returns:
        list: List of paragraphs
    """
    import re
    
    # First, replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Split text into paragraphs based on newlines
    paragraphs = re.split(r'\n+', text)
    
    # Remove any empty paragraphs and strip whitespace from each paragraph
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs

def validate_output(outs):
    i=0
    validated_outs = []
    validated_outs_thread_ids = []
    prevalidated_outs = []
    non_validated_outs = []
    non_object_outs = []
    for out in outs:
        if out.json_object:
            try:
                validated_outs.append(NarrativeAnalysis.model_validate(out.json_object.object))
                validated_outs_thread_ids.append(out.chat_thread_live_id)
                prevalidated_outs.append(out)
            except Exception as e:
                non_validated_outs.append(out)
                print(e)
            i=i+1
        else:
            non_object_outs.append(out)
    print(i,len(validated_outs),len(validated_outs_thread_ids),len(prevalidated_outs),len(non_validated_outs),len(non_object_outs),len(outs))
    return validated_outs,validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs
import os
async def main(book_id: int,chunks_size: int=2000, max_calls: Optional[int] = None, base_path: str = "/Users/tommasofurlanello/Documents/Dev/MarketInference/data/"):
    action_extractor = StructuredTool.from_pydantic(NarrativeAnalysis)
    action_extractor.post_validate_schema = False

    system_prompt = SystemPrompt(name="Narrative Action Extraction System", content=system_string)

    llm_config_vllm_modal = LLMConfig(client=LLMClient.vllm, model=vllm_model, response_format=ResponseFormat.structured_output,max_tokens=6000)
    data_name = "gutenberg_en_novels.parquet"
    joined_data_path = os.path.join(base_path, data_name)
    try:
        novels = pl.read_parquet(joined_data_path)
    except Exception as e:
        print(f"Error reading parquet file: {e} did you remember to update the url with your own path?")
        novels = pl.read_csv(joined_data_path)

    book_str = novels["TEXT"][book_id]

    book_paragraphs = clean_gutenberg_text(book_str)
    book_paragraphs_str = "\n".join(book_paragraphs)
    print(len(book_str),len(book_paragraphs_str))

    book_paragraphs_df = pl.DataFrame({"TEXT": book_paragraphs}).with_columns(pl.col("TEXT").str.len_chars().alias("LENGTH"))

    paragraphs = {0:[]}
    paragraphs_lengths = {0:0}
    paragraph_id = 0
    max_length = chunks_size
    for text,length in book_paragraphs_df.iter_rows():
        if paragraphs_lengths[paragraph_id] + length > max_length:
            paragraph_id += 1
            paragraphs[paragraph_id] = []
            paragraphs_lengths[paragraph_id] = 0
        paragraphs[paragraph_id].append(text)
        paragraphs_lengths[paragraph_id] += length

    paragraphs_strings = [ "\n".join(paragraphs[i]) for i in paragraphs.keys()]

    paragraphs_df = pl.DataFrame({"TEXT": paragraphs_strings}).with_columns(pl.col("TEXT").str.len_chars().alias("LENGTH"))
        
    chunks_number_context = 0
    i=10
    prompts = []
    chunk_list = []
    context_list = []
    for i, chunk in enumerate(paragraphs_df["TEXT"]):
        context_chunks_start = max(0, i - chunks_number_context)
        non_inclusive_context_str = paragraphs_df["TEXT"][context_chunks_start:i].str.concat(delimiter="\n")[0]
        target_chunk = chunk
        prompt = f""" Utilize the context in the following text {non_inclusive_context_str} to better understand the target text {target_chunk} and extract the actions in the target text."""
        prompts.append(prompt)
        chunk_list.append(target_chunk)
        context_list.append(non_inclusive_context_str)

    prompts_df = pl.DataFrame({"prompt": prompts, "chunk": chunk_list, "context": context_list})
    if max_calls and len(prompts_df) > max_calls:
        example_prompts = prompts_df.head(max_calls)
    else:
        example_prompts = prompts_df


    threads, prompts_df_with_thread_ids = create_vllm_threads(example_prompts, system_prompt, llm_config_vllm_modal, action_extractor)

    outs = await orchestrator.run_parallel_ai_completion(threads)

    # #save outs to file
    # with open(f"/Users/tommasofurlanello/Documents/Dev/MarketInference/data/gutenberg_en_novels_actions_{book_id}.json", "w") as f:
    #     json.dump(outs, f)





    validated_outs, validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs = validate_output(outs)

    outs_with_actions = []
    outs_with_actions_thread_ids = []
    for out, out_ids in zip(validated_outs, validated_outs_thread_ids):
        if out.text_had_no_actions == False:
            outs_with_actions.append(out)
            outs_with_actions_thread_ids.append(str(out_ids))
            

    outs_frame = pl.DataFrame(outs_with_actions)
    outs_id_frame = pl.DataFrame({"thread_id": outs_with_actions_thread_ids})
    outs_frame_with_ids = pl.concat([outs_frame, outs_id_frame], how="horizontal")
    print(outs_frame_with_ids)
    print(prompts_df_with_thread_ids)
    out_sframe_with_ids_joined_prompts = outs_frame_with_ids.join(prompts_df_with_thread_ids, on="thread_id", how="left").sort("index")
    out_name = f"gutenberg_en_novels_actions_{book_id}_{chunks_size}.parquet"
    out_path = os.path.join(base_path, out_name)

    out_sframe_with_ids_joined_prompts.write_parquet(out_path)
    print(f"outs_frame saved to {out_path}")
    print(out_sframe_with_ids_joined_prompts)
    



if __name__ == "__main__":
    start = time.time()
    asyncio.run(main(book_id=0,chunks_size=2000,max_calls=None))
    end = time.time()
    print(f"Time taken: {end - start} seconds")