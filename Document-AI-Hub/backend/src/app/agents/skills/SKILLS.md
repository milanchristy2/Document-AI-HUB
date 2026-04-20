# Skills Instructions for RAG Agent

## Overview
Skills are reusable capabilities that agents can invoke to perform specific tasks. Each skill handles a particular aspect of the RAG pipeline.

## Core Skills

### 1. Retrieve Skill
**Purpose**: Document retrieval and search

**Inputs**:
- `query` (str): The search query
- `top_k` (int): Number of documents to retrieve (default: 6)
- `document_id` (Optional[str]): Filter by specific document
- `strategy` (str): Retrieval strategy - "standard", "summary", "extraction", "comparison"

**Outputs**:
- `chunks` (List[Dict]): Retrieved document chunks with scores
- `total_count` (int): Total matches found
- `retrieval_time` (float): Execution time in ms

**Logic**:
1. Validate and expand query using HybridRetriever
2. Execute retrieval based on strategy
3. Apply document filtering if document_id provided
4. Return top_k results with metadata

---

### 2. Augment Skill
**Purpose**: Context augmentation and RAG

**Inputs**:
- `query` (str): Original query
- `chunks` (List[Dict]): Retrieved chunks
- `rerank` (bool): Whether to rerank results

**Outputs**:
- `context` (str): Augmented context text
- `reranked_chunks` (List[Dict]): Reranked chunks
- `augmentation_score` (float): Quality score

**Logic**:
1. Combine chunks into coherent context
2. Apply reranking if enabled
3. Filter low-quality or redundant chunks
4. Return augmented context with source attribution

---

### 3. Synthesize Skill
**Purpose**: LLM-based answer generation

**Inputs**:
- `query` (str): User query
- `context` (str): Augmented context
- `user_role` (Optional[str]): User role for personalization

**Outputs**:
- `answer` (str): Generated answer
- `confidence` (float): Confidence score 0-1
- `sources` (List[str]): Source references
- `thinking_process` (str): Chain of thought

**Logic**:
1. Build system prompt based on context
2. Call LLM with context and query
3. Verify answer against evidence
4. Calculate confidence based on evidence alignment
5. Return answer with metadata

---

## Skill Implementation Pattern

```python
from app.agents.base_agent import BaseSkill, SkillResult

class MySkill(BaseSkill):
    def __init__(self):
        super().__init__(name="my_skill", description="Does something")
    
    async def validate_input(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate input parameters"""
        required = ["query"]
        for param in required:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"
        return True, None
    
    async def execute(self, **kwargs) -> SkillResult:
        """Main execution logic"""
        try:
            # Your logic here
            result = "success"
            return SkillResult(success=True, data={"result": result})
        except Exception as e:
            return SkillResult(success=False, error=str(e))
    
    async def pre_execute(self, **kwargs) -> None:
        """Setup before execution"""
        pass
    
    async def post_execute(self, result: SkillResult) -> SkillResult:
        """Cleanup after execution"""
        return result
```

---

## Adding New Skills

1. **Define the Skill Class**
   - Extend BaseSkill
   - Implement `validate_input()`
   - Implement `execute()`
   - Override `pre_execute()` / `post_execute()` if needed

2. **Register with Agent**
   ```python
   agent = MyAgent(config)
   agent.register_skill(my_skill)
   ```

3. **Call from Agent**
   ```python
   result = await agent.call_skill("my_skill", param1=value1, param2=value2)
   ```

---

## Error Handling

All skills should:
- Return `SkillResult(success=False, error="message")` on failure
- Log errors with context
- Not raise exceptions (handle internally)
- Provide meaningful error messages

---

## Performance Considerations

- Skills are executed in sequence by default
- Each skill tracks execution time
- Statistics are automatically collected
- Retry logic with exponential backoff (2^attempt seconds)
- Default timeout: 30 seconds per skill execution

---

## Example: Retrieve Skill Usage

```python
# In RAG Agent _act method
result = await self.call_skill(
    "retrieve",
    query="What is machine learning?",
    top_k=6,
    strategy="standard"
)

if result.success:
    chunks = result.data["chunks"]
    for chunk in chunks:
        print(f"Score: {chunk['score']}, Content: {chunk['text']}")
else:
    print(f"Error: {result.error}")
```

---

## Monitoring

Check skill performance:

```python
agent = MyAgent(config)
summary = agent.get_skills_summary()
# Returns:
# {
#     "retrieve": {
#         "executions": 10,
#         "success_rate": 95.0,
#         "avg_time_ms": 245.5
#     },
#     ...
# }
```

---

## Built-in Skills in This System

1. **retrieve**: Hybrid vector + keyword search
2. **augment**: Context augmentation with reranking
3. **synthesize**: LLM-based answer generation

---

## Custom Skill Examples

See `/app/skills/` directory for implementations:
- `retrieval_skill.py`: Document retrieval
- `augmentation_skill.py`: Context augmentation
- `synthesis_skill.py`: Answer generation
