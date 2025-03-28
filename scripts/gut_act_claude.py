import asyncio
from dotenv import load_dotenv
from minference.threads.inference import InferenceOrchestrator, RequestLimits
from minference.threads.models import ChatMessage, ChatThread, LLMConfig, CallableTool, LLMClient, ResponseFormat, SystemPrompt, StructuredTool, Usage, GeneratedJsonObject
from typing import Literal, List, Optional, Dict, Any
from minference.ecs.caregistry import CallableRegistry
import time
from minference.clients.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
from minference.ecs.entity import EntityRegistry
import os
import logging
import json
import polars as pl
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("book_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("book_processor")

# Load environment variables
load_dotenv()
EntityRegistry()
CallableRegistry()

# Configure the orchestrator
vllm_request_limits = RequestLimits(max_requests_per_minute=16, max_tokens_per_minute=200000000)
vllm_model = "Qwen/Qwen2.5-32B-Instruct"
vllm_endpoint_autoscale = ""
orchestrator = InferenceOrchestrator(vllm_request_limits=vllm_request_limits, vllm_endpoint=vllm_endpoint_autoscale)
EntityRegistry.set_inference_orchestrator(orchestrator)
EntityRegistry.set_tracing_enabled(False)

# Narrative Action Extraction System
system_string = """
You are an expert system designed to extract structured actions from narrative text. Your primary goal is to identify and extract actions, like interactions, movements, observable behaviors and mental changes from narratives, even when they might be subtle or implied. Always prioritize finding actions rather than concluding none exist.

## IMPORTANT: Action Extraction Is Your Primary Task

The most important part of your analysis is extracting actions between entities. **ALWAYS thoroughly search for actions in the text before concluding none exist.** Consider these types of interactions as valid actions:

1. Direct physical interactions (e.g., "Maya picked up the lantern")
2. Movements (e.g., "The fox darted into the forest")
3. Observable behaviors (e.g., "Professor Lin frowned", "Raj smiled")
4. Implied physical actions (e.g., "Sarah found herself tumbling down the hillside" implies the action "tumble")
5. Actions described in dialogue (e.g., "'I tossed it over the fence,' said Eliza" implies the action "toss")
6. Mental changes (e.g., "She realized the truth" implies the action "realize")
7. Emotions (e.g., "She felt sad" implies the action "feel")

**Do not be overly strict in what qualifies as an action.** If there is any observable behavior or physical movement in the text, it should be captured as an action.

## Action Extraction Process

1. **First, carefully read the text and list all potential actions** - be generous in what you consider an action
2. For each potential action:
   - Identify source entity (who/what performs the action)
   - Identify target entity (who/what receives the action)
   - Extract the verb describing the action
   - Determine the consequence of the action
   - Note the text evidence supporting the action
   - Assign a location and temporal order

## NarrativeAnalysis Model Structure

The NarrativeAnalysis model contains:
- `text_id`: A unique identifier for the analyzed text segment
- `text_had_no_actions`: Boolean indicating whether the text contained actions (default to FALSE)
- `actions`: List of Action objects, ordered by temporal sequence

## Action Model Structure

Each Action object contains:
- `source`: Name of the entity performing the action
- `source_type`: Category of the source (person, animal, object, location)
- `source_is_character`: Whether the source is a named character
- `target`: Name of the entity receiving the action
- `target_type`: Category of the target (person, animal, object, location)
- `target_is_character`: Whether the target is a named character
- `action`: The verb or short phrase describing the physical interaction
- `consequence`: The immediate outcome or result of the action
- `text_describing_the_action`: Text fragment describing the action
- `text_describing_the_consequence`: Description of the consequence
- `location`: location of the action
- `temporal_order_id`: Sequential identifier for chronological order

## IMPORTANT: Handling Repeated Actions

When the same action verb appears multiple times in the narrative (e.g., "walk" happening at different moments), create separate Action objects for each occurrence with:
1. Different `temporal_order_id` values reflecting their sequence
2. Appropriate description of each specific instance
3. Proper placement in the actions list according to chronological order

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
  "actions": [
    {
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
      "location": "cave",
      "temporal_order_id": 1
    },
    {
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
      "location": "cave",
      "temporal_order_id": 2
    },
    {
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
      "location": "cave",
      "temporal_order_id": 3
    },
    {
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
      "location": "cave",
      "temporal_order_id": 4
    },
    {
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
      "location": "cave",
      "temporal_order_id": 5
    },
    {
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
      "location": "cave",
      "temporal_order_id": 6
    },
    {
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
      "location": "cave",
      "temporal_order_id": 7
    },
    {
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
      "location": "cave",
      "temporal_order_id": 8
    },
    {
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
      "location": "cave",
      "temporal_order_id": 9
    },
    {
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
      "location": "cave",
      "temporal_order_id": 10
    },
    {
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
      "location": "cave",
      "temporal_order_id": 11
    }
  ]
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
  "actions": [
    {
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
      "location": "desk,
      "temporal_order_id": 1
    },
    {
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
      "location": "desk,
      "temporal_order_id": 2
    },
    {
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
      "location": "desk,
      "temporal_order_id": 3
    },
    {
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
      "location": "desk,
      "temporal_order_id": 4
    },
    {
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
      "location": "desk,
      "temporal_order_id": 5
    },
    {
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
      "location": "desk,
      "temporal_order_id": 6
    },
    {
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
      "location": "doorway",
      "temporal_order_id": 7
    },
    {
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
      "location":  "doorway",
      "temporal_order_id": 8
    },
    {
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
      "location": "desk",
      "temporal_order_id": 9
    }
  ]
}
```

## Example with Repeated Actions

For the text:
"The dog walked to the door. It barked loudly, its tail wagging excitedly. The owner walked to the door and opened it. The dog walked outside, still wagging its tail."

The NarrativeAnalysis would include two separate instances of "walk" for the dog and one for the owner, and two instances of "wag" for the tail:

```json
{
  "text_id": "dog-owner-door",
  "text_had_no_actions": false,
  "actions": [
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The dog moves to the door",
      "text_describing_the_action": "The dog walked to the door",
      "text_describing_the_consequence": "The dog is now at the door",
      "location": "doorway",
      "temporal_order_id": 1
    },
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "house",
      "target_type": "location",
      "target_is_character": false,
      "action": "bark",
      "consequence": "Sound is produced",
      "text_describing_the_action": "It barked loudly",
      "text_describing_the_consequence": "Barking sound is heard",
      "location": "doorway",
      "temporal_order_id": 2
    },
    {
      "source": "tail",
      "source_type": "object",
      "source_is_character": false,
      "target": "dog",
      "target_type": "animal",
      "target_is_character": true,
      "action": "wag",
      "consequence": "The tail moves back and forth",
      "text_describing_the_action": "its tail wagging excitedly",
      "text_describing_the_consequence": "The dog's tail is in motion showing excitement",
      "location": "doorway",
      "temporal_order_id": 3
    },
    {
      "source": "owner",
      "source_type": "person",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The owner moves to the door",
      "text_describing_the_action": "The owner walked to the door",
      "text_describing_the_consequence": "The owner is now at the door",
      "location": "doorway",
      "temporal_order_id": 4
    },
    {
      "source": "owner",
      "source_type": "person",
      "source_is_character": true,
      "target": "door",
      "target_type": "object",
      "target_is_character": false,
      "action": "open",
      "consequence": "The door changes from closed to open",
      "text_describing_the_action": "and opened it",
      "text_describing_the_consequence": "The door is now open",
      "location": "doorway",
      "temporal_order_id": 5
    },
    {
      "source": "dog",
      "source_type": "animal",
      "source_is_character": true,
      "target": "outside",
      "target_type": "location",
      "target_is_character": false,
      "action": "walk",
      "consequence": "The dog moves from inside to outside",
      "text_describing_the_action": "The dog walked outside",
      "text_describing_the_consequence": "The dog is now outside the house",
      "location":  "doorway",
      "temporal_order_id": 6
    },
    {
      "source": "tail",
      "source_type": "object",
      "source_is_character": false,
      "target": "dog",
      "target_type": "animal",
      "target_is_character": true,
      "action": "wag",
      "consequence": "The tail continues to move back and forth",
      "text_describing_the_action": "still wagging its tail",
      "text_describing_the_consequence": "The dog's tail continues to show excitement",
      "location":  "doorway",
      "temporal_order_id": 7
    }
  ]
}


## Required Output Format

Always return a complete NarrativeAnalysis object following the provided schema. Ensure the output can be parsed as JSON without errors. Do not include any nested tool_call tags or extra formatting in your response.

Remember, your primary task is to extract ALL possible actions from the text, even subtle ones. Each action should be represented as an Action object in the `actions` list, properly ordered by temporal sequence.

Pay special attention to:
1. Repeated actions (same verb) occurring at different times
2. Subtle actions like expressions, gestures, or sensory perceptions
3. Implied actions that are not explicitly stated
4. Actions described in dialogue
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
    text_describing_the_action: str = Field(..., description="Text fragment describing the action exactly as it is written in the text")
    text_describing_the_consequence: str = Field(..., description="Description of the consequence exactly as it is written in the text")
    
    # Context information
    location: str = Field(..., description="location from global to local")
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
    
    # Actions indexed by name
    actions: List[Action] = Field(
        default_factory=list,
        description="Dictionary mapping action names to Action objects"
    )

def clean_lines(lines: str) -> list:
    """Clean text by removing excessive whitespace and empty lines."""
    import re
    split_lines = lines.split("\n\r\n\r\n")
    newlines = []
    for line in split_lines: 
        l = re.sub('(\r\n)+\r?\n?', ' ', line)
        l = re.sub(r"\s\s+", " ", l)
        if not re.search(r'^[^a-zA-Z0-9]+$', l):
            newlines.append(l.strip())
    return newlines

def create_vllm_threads(prompts_df: pl.DataFrame, system_prompt: SystemPrompt, 
                       llm_config_vllm_modal: LLMConfig, tool: StructuredTool) -> tuple:
    """Create ChatThread objects for each prompt."""
    if not isinstance(prompts_df, pl.DataFrame):
        raise ValueError("prompts_df must be a pl.DataFrame")
    if "prompt" not in prompts_df.columns:
        raise ValueError("prompts_df must contain a 'prompt' column")
    
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
    
    # Add thread IDs to the DataFrame
    thread_id = [str(thread.live_id) for thread in vllm_threads]
    prompts_df_with_thread_ids = prompts_df.with_columns(
        pl.Series(name="thread_id", values=thread_id)
    ).with_row_index()
    
    return vllm_threads, prompts_df_with_thread_ids

def validate_output(outs: list) -> tuple:
    """Validate the output from the LLM."""
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
                logger.warning(f"Validation error: {e}")
                non_validated_outs.append(out)
        else:
            non_object_outs.append(out)
    
    logger.info(f"Validation results: {len(validated_outs)} validated, {len(non_validated_outs)} not validated, {len(non_object_outs)} not objects")
    return validated_outs, validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs

async def process_book_chunks(book_str: str, book_id: int, chunks_size: int = 2000, 
                             batch_size: int = 8, base_path: str = "./data/") -> Dict[str, Any]:
    """
    Process a single book with consistent batch sizes and graceful error handling.
    
    Each book is processed independently, and chunks are sent in sequential batches
    to maintain a consistent request rate.
    """
    start_time = time.time()
    out_name = f"gutenberg_en_novels_actions_compact_{book_id}_{chunks_size}.parquet"
    out_path = os.path.join(base_path, out_name)
    
    # Skip if file already exists
    if os.path.exists(out_path):
        logger.info(f"Book {book_id}: File {out_name} already exists, skipping...")
        return {"book_id": book_id, "status": "skipped", "duration": 0}
    
    # Initialize results storage
    all_outs = []
    all_thread_ids = []
    all_batches_processed = True
    
    try:
        # Preprocess book text
        book_paragraphs = clean_lines(book_str)
        if not book_paragraphs:
            logger.warning(f"Book {book_id}: No valid paragraphs found")
            return {"book_id": book_id, "status": "no_paragraphs", "duration": time.time() - start_time}
        
        # Create chunks of text
        paragraphs = {0: []}
        paragraphs_lengths = {0: 0}
        paragraph_id = 0
        
        # Create DataFrame of paragraphs
        book_paragraphs_df = pl.DataFrame({"TEXT": book_paragraphs}).with_columns(
            pl.col("TEXT").str.len_chars().alias("LENGTH")
        )
        
        for text, length in book_paragraphs_df.iter_rows():
            if paragraphs_lengths[paragraph_id] + length > chunks_size:
                paragraph_id += 1
                paragraphs[paragraph_id] = []
                paragraphs_lengths[paragraph_id] = 0
            paragraphs[paragraph_id].append(text)
            paragraphs_lengths[paragraph_id] += length
        
        paragraphs_strings = ["\n".join(paragraphs[i]) for i in paragraphs.keys()]
        paragraphs_df = pl.DataFrame({"TEXT": paragraphs_strings}).with_columns(
            pl.col("TEXT").str.len_chars().alias("LENGTH")
        )
        
        # Skip books with fewer than 5 chunks
        if len(paragraphs_strings) < 5:
            logger.info(f"Book {book_id}: Only {len(paragraphs_strings)} chunks, skipping...")
            return {"book_id": book_id, "status": "too_few_chunks", "duration": time.time() - start_time}
        
        # Create prompts for each chunk
        chunks_number_context = 0
        prompts = []
        chunk_list = []
        context_list = []
        
        for i, chunk in enumerate(paragraphs_df["TEXT"]):
            context_chunks_start = max(0, i - chunks_number_context)
            non_inclusive_context_str = paragraphs_df["TEXT"][context_chunks_start:i].str.concat(delimiter="\n")[0]
            target_chunk = chunk
            prompt = f"""Utilize the context in the following text {non_inclusive_context_str} to better understand the target text {target_chunk} and extract the actions in the target text."""
            prompts.append(prompt)
            chunk_list.append(target_chunk)
            context_list.append(non_inclusive_context_str)
        
        prompts_df = pl.DataFrame({"prompt": prompts, "chunk": chunk_list, "context": context_list})
        
        # Initialize tools and configs
        action_extractor = StructuredTool.from_pydantic(NarrativeAnalysis)
        action_extractor.post_validate_schema = False
        system_prompt = SystemPrompt(name="Narrative Action Extraction System", content=system_string)
        llm_config_vllm_modal = LLMConfig(
            client=LLMClient.vllm,
            model=vllm_model,
            response_format=ResponseFormat.structured_output,
            max_tokens=6000
        )
        
        # Process in consistent batch sizes
        total_batches = (len(prompts_df) + batch_size - 1) // batch_size
        logger.info(f"Book {book_id}: Processing {len(prompts_df)} chunks in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts_df))
            batch_df = prompts_df.slice(start_idx, end_idx - start_idx)
            
            try:
                # Create threads for this batch
                batch_threads, batch_df_with_ids = create_vllm_threads(
                    batch_df, system_prompt, llm_config_vllm_modal, action_extractor
                )
                
                # Process this batch with timeout
                logger.info(f"Book {book_id}: Processing batch {batch_idx+1}/{total_batches} with {len(batch_threads)} threads")
                try:
                    batch_results = await asyncio.wait_for(
                        orchestrator.run_parallel_ai_completion(batch_threads),
                        timeout=300  # 5 minute timeout per batch
                    )
                    all_outs.extend(batch_results)
                    logger.info(f"Book {book_id}: Batch {batch_idx+1}/{total_batches} completed successfully")
                except asyncio.TimeoutError:
                    logger.warning(f"Book {book_id}: Timeout processing batch {batch_idx+1}/{total_batches}")
                    all_batches_processed = False
                    continue
                except Exception as e:
                    logger.warning(f"Book {book_id}: Error processing batch {batch_idx+1}/{total_batches}: {e}")
                    all_batches_processed = False
                    continue
                
                # Add small delay between batches to avoid overwhelming the API
                await asyncio.sleep(1)
                
            except Exception as batch_e:
                logger.warning(f"Book {book_id}: Error preparing batch {batch_idx+1}/{total_batches}: {batch_e}")
                all_batches_processed = False
                continue
        
        # If we have no results at all, return early
        if not all_outs:
            logger.warning(f"Book {book_id}: No successful results from any batch")
            return {"book_id": book_id, "status": "no_results", "duration": time.time() - start_time}
        
        # Validate and process results
        validated_outs, validated_outs_thread_ids, prevalidated_outs, non_validated_outs, non_object_outs = validate_output(all_outs)
        
        # Extract actions
        outs_with_actions = []
        outs_with_actions_thread_ids = []
        token_usage = []
        
        for out, out_ids, preout in zip(validated_outs, validated_outs_thread_ids, prevalidated_outs):
            if not out.text_had_no_actions:
                outs_with_actions.append(out)
                outs_with_actions_thread_ids.append(str(out_ids))
                token_usage.append(preout.usage.completion_tokens)
        
        # Handle the case with no actions found
        if not outs_with_actions:
            logger.warning(f"Book {book_id}: No actions found in any valid output")
            
            # Create an empty DataFrame with proper schema and save it
            empty_df = pl.DataFrame({
                "text_id": [], "text_had_no_actions": [], "thread_id": [],
                "token_usage": [], "index": [], "prompt": [], "chunk": [], "context": []
            })
            empty_df.write_parquet(out_path)
            
            return {
                "book_id": book_id, 
                "status": "no_actions", 
                "all_batches_processed": all_batches_processed,
                "duration": time.time() - start_time
            }
        
        # Save results
        try:
            # Create DataFrames
            outs_frame = pl.DataFrame(outs_with_actions)
            outs_id_frame = pl.DataFrame({
                "thread_id": outs_with_actions_thread_ids,
                "token_usage": token_usage
            })
            
            # Join DataFrames
            outs_frame_with_ids = pl.concat([outs_frame, outs_id_frame], how="horizontal")
            
            # Create a DataFrame with all thread IDs from prompts
            all_threads_df = pl.concat([batch_df_with_ids for batch_idx in range(total_batches) 
                                      if batch_idx * batch_size < len(prompts_df)])
            
            # Join with prompts data
            out_frame_joined = outs_frame_with_ids.join(
                all_threads_df, on="thread_id", how="left"
            )
            
            # Sort by index if it exists
            if "index" in out_frame_joined.columns:
                out_frame_joined = out_frame_joined.sort("index")
            
            # Save to parquet
            out_frame_joined.write_parquet(out_path)
            logger.info(f"Book {book_id}: Successfully saved results to {out_path}")
            
            return {
                "book_id": book_id, 
                "status": "success", 
                "all_batches_processed": all_batches_processed,
                "actions_found": len(outs_with_actions),
                "duration": time.time() - start_time
            }
            
        except Exception as save_e:
            logger.error(f"Book {book_id}: Error saving results: {save_e}")
            
            # Try to save partial results
            try:
                partial_df = pl.DataFrame({"thread_id": outs_with_actions_thread_ids, "book_id": book_id})
                partial_path = os.path.join(base_path, f"partial_{book_id}_{chunks_size}.parquet")
                partial_df.write_parquet(partial_path)
                logger.info(f"Book {book_id}: Saved partial results to {partial_path}")
            except Exception as partial_e:
                logger.error(f"Book {book_id}: Error saving partial results: {partial_e}")
            
            return {
                "book_id": book_id, 
                "status": "save_error", 
                "error": str(save_e),
                "duration": time.time() - start_time
            }
    
    except Exception as e:
        logger.error(f"Book {book_id}: Unexpected error: {e}")
        return {
            "book_id": book_id, 
            "status": "error", 
            "error": str(e),
            "duration": time.time() - start_time
        }

async def process_books_independently(book_ids: list, max_concurrent: int = 10, 
                                     chunks_size: int = 2000, batch_size: int = 8,
                                     base_path: str = "./data/"):
    """
    Process multiple books independently with controlled concurrency.
    
    Each book is completely independent, and we use a semaphore to limit
    how many books are processed concurrently.
    """
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    data_name = "gutenberg_en_novels.parquet"
    joined_data_path = os.path.join(base_path, data_name)
    
    try:
        novels = pl.read_parquet(joined_data_path)
    except Exception as e:
        logger.error(f"Error reading novels data: {e}")
        return []
    
    async def process_book_with_semaphore(book_id):
        async with semaphore:
            try:
                book_str = novels["TEXT"][book_id]
                result = await process_book_chunks(
                    book_str, book_id, chunks_size, batch_size, base_path
                )
                return result
            except Exception as e:
                logger.error(f"Error processing book {book_id}: {e}")
                return {"book_id": book_id, "status": "error", "error": str(e)}
    
    # Create tasks for all books
    tasks = [process_book_with_semaphore(book_id) for book_id in book_ids]
    
    # Process all books and collect results as they complete
    for completed_task in asyncio.as_completed(tasks):
        try:
            result = await completed_task
            results.append(result)
            logger.info(f"Completed book {result['book_id']} with status: {result['status']}")
        except Exception as e:
            logger.error(f"Error awaiting task: {e}")
    
    return results

async def main():
    """Main entry point for the script."""
    start_time = time.time()
    
    # Configuration
    book_start = 0
    num_books = 500
    max_concurrent_books = 24  # Maximum number of books to process concurrently
    chunks_size = 2000
    batch_size = 8  # Number of chunks to process in each batch
    base_path = "/Users/tommasofurlanello/Documents/Dev/MarketInference/data/"
    
    # Generate list of book IDs to process
    book_ids = list(range(book_start, book_start + num_books))
    
    # Process books independently
    logger.info(f"Starting processing of {len(book_ids)} books with max {max_concurrent_books} concurrent")
    results = await process_books_independently(
        book_ids, max_concurrent_books, chunks_size, batch_size, base_path
    )
    
    # Summarize results
    status_counts = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info(f"Processing completed in {total_duration:.2f} seconds")
    logger.info(f"Status summary: {status_counts}")
    
    # Save results summary
    try:
        results_df = pl.DataFrame(results)
        summary_path = os.path.join(base_path, f"processing_summary_{int(start_time)}.parquet")
        results_df.write_parquet(summary_path)
        logger.info(f"Results summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Error saving results summary: {e}")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())