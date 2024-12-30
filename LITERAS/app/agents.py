
import json
from typing import Dict, List
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from .tools import pubmed_search

class AcademicSearchTeam:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        # model client
        self.start_time = None
        self.refine_search_count = 0
        self.proceed_to_synthesis_count = 0
        self.validator_scores = []
        self.total_studies = 0
        self.reference_validation_count = 0          
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            temperature=0.7,
            api_key=api_key
        )

        #tools
        self.pubmed_tool = FunctionTool(
            pubmed_search,
            description="Search PubMed for academic articles and return structured results"
        )

        #agents
        self.query_planner = AssistantAgent(
            name="QueryPlanner",
            model_client=self.model_client,
            description="Expert at generating comprehensive search queries",
            system_message="""You are an expert at developing comprehensive academic search strategies.
            For any given research topic:
            1. Break down the topic into key concepts and subthemes
            2. Generate multiple search queries considering:
               - Core terminology and synonyms
               - Related concepts and methodologies
               - Different theoretical frameworks
               - Various applications and contexts
               - Field-specific terminology
            3. Structure queries from specific to broader related topics
            4. Include both narrow, focused queries and broader, contextual ones
            5. Consider temporal aspects (recent developments vs historical context)
            
            Format your response as a JSON object with these fields:
            {
                "main_queries": ["list of primary topic queries"]
            }
            
            Aim to generate 3-5 queries in the category.
            If receiving refinement request, modify queries based on feedback."""
        )

        self.search_agent = AssistantAgent(
            name="SearchAgent",
            model_client=self.model_client,
            tools=[self.pubmed_tool],
            description="Academic literature search execution specialist",
            system_message="""You are an expert at executing comprehensive academic searches.
            For each query provided:
            1. Execute search using pubmed_search tool
            2. Track which queries yielded the most relevant results
            3. Remove duplicate results across queries
            4. Maintain a count of total unique articles found
            5. For each successful query, report:
               - Number of results found
               - Brief assessment of result relevance
               - Any suggested query refinements"""
        )

        self.validator_agent = AssistantAgent(
            name="Validator",
            model_client=self.model_client,
            description="Paper scoring specialist",
            system_message="""Score each article using these criteria (total 25 points):
            1. Direct relevance to the research topic (0-5 points)
            2. Recency of publication (0-5 points)
            3. Study type/methodology (0-5 points)
            4. Clinical applicability (0-5 points)
            5. Innovation/novelty (0-5 points)
            
            Format output as:
            {
                "scored_papers": [
                    {
                        "title": "paper title",
                        "doi": "doi number",
                        "relevance_score": X,
                        "recency_score": Y,
                        "methodology_score": Z,
                        "applicability_score": A,
                        "innovation_score": B,
                        "total_score": N,
                        "reason": "explanation"
                    }
                ],
                "summary": {
                    "total_papers": N,
                    "high_quality_papers": M  # papers scoring ≥20 total
                }
            }"""
        )

        self.critic_agent = AssistantAgent(
            name="Critic",
            model_client=self.model_client,
            description="Search and validation quality critic",
            system_message="""Evaluate both search results and validation scores.

            When reviewing SEARCH OUTPUT:
            1. Check number of results (minimum 10 papers needed)
            2. Check coverage of topic aspects
            If insufficient:
            - State "REFINE_SEARCH"
            - Explain what aspects need more coverage

            When reviewing VALIDATION SCORES:
            1. Check if at least 3 papers score ≥20 total points
            2. Verify papers truly match research question
            
            If ANY requirement not met:
            - Output "REFINE_SEARCH"
            - List specific issues to address
            
            If ALL requirements met:
            - Output "PROCEED_TO_SYNTHESIS"
            Approved References:
            [
                {
                    "title": "exact paper title",
                    "authors": ["author1", "author2", "et al."],
                    "year": "YYYY",
                    "journal": "full journal name",
                    "doi": "doi number",
                    "citation_key": "Author2024" // FirstAuthorYear format
                },
                // ... other papers
            ]
            
            Final selected papers: (for human reading)
            1. Title, Authors, DOI
            2. ...
            ```
            
            IMPORTANT: 
            - Never output both REFINE_SEARCH and PROCEED_TO_SYNTHESIS
            - Always include complete reference details in JSON format
            - Use consistent citation keys for paper identification"""

        )

        
        self.synthesis_agent = AssistantAgent(
            name="SynthesisAgent",
            model_client=self.model_client,
            description="Medical research paper introduction specialist",
            system_message="""You are an expert at writing medical research paper introductions with proper citations.
            IMPORTANT: Only use references that have been validated by the Critic agent.
                        CRITICAL REQUIREMENTS:
            1. Use ONLY references explicitly approved by critic
            2. NO external citations (e.g., WHO, general statistics)
            3. ALL claims must be supported by approved references
            4. NO general statements without specific citations
            
            Create a structured introduction following this format:

            1. Topic Introduction:
               - Provide clear context and overview
               - Use relevant statistics and background information
               - Each claim must be supported by citations

            2. Literature Summary:
               - Summarize current knowledge and theories
               - Present key findings from major studies
               - Organize information logically
               - Every study mentioned must be cited

            3. Gap Identification:
               - Clearly state what's missing in current research
               - Explain why this gap is important
               - Support gap identification with citations

            4. Study Objective:
               - State the specific aims clearly
               - Connect objectives to identified gaps
               - Explain potential impact
               
             5. References:
                - Include all references used in the synthesis in the same format as critic-approved papers
                  
               
            After completion:
            State "SYNTHESIS_COMPLETE\""""
        )
        
        #reference consistency critic
        self.reference_consistency_critic = AssistantAgent(
            name="ReferenceConsistencyCritic",
            model_client=self.model_client,
            description="Validates synthesis references against approved list",
            system_message="""Validate that synthesis ONLY uses approved references.

            CHECK EACH:
            1. Extract all citations from synthesis text
            2. Compare against approved_references from critic agent
            3. Verify:
               - Citation format matches citation_key
               - Claims align with cited paper content
               
            Output Format if ANY issues:
            ```
            REVISE_NEEDED

            Invalid Citations:
            1. Line: [text with citation]
               Found: [current citation]
               Required: [correct citation_key]
               
            Available Approved References:
            [List citation_keys and details]

            Revise synthesized text to use correct citations.
            ```

            Output if all valid:
            ```
            PROCEED_TO_FORMATTING

            Validation Summary:
            - Total citations: [number]
            - All citations match approved keys
            - Citation contexts verified
            ```"""
        )

        self.formatter_agent = AssistantAgent(
            name="FormatterAgent",
            model_client=self.model_client,
            description="Medical paper formatter specialist",
            system_message="""Format the final paper in markdown:
            
            # [Paper Title]
            
            ## Introduction
            [Copy introduction sections from synthesis]
            
            ## References
            
            1. Author(s). Title. *Journal*. Year;Volume(Issue):Pages. DOI
            2. [Continue reference list]
            
            State '**TERMINATE**' when complete."""
        )
        
        self.current_phase = "SEARCH"

        def selector_func(messages):
            if not messages:
                self.start_time = time.time()
                return "QueryPlanner"
            
            last_message = messages[-1]
            content = str(last_message.content).upper()

            # Track metrics - keep existing tracking
            if last_message.source == "Critic":
                if "REFINE_SEARCH" in content:
                    self.refine_search_count += 1
                if "PROCEED_TO_SYNTHESIS" in content:
                    self.proceed_to_synthesis_count += 1
                    
            if last_message.source == "Validator":
                try:
                    scores = json.loads(content)
                    if "scored_papers" in scores:
                        self.validator_scores.extend(scores["scored_papers"])
                        self.total_studies = len(scores["scored_papers"])
                except:
                    pass

            # Selection logic with reference validation
            if self.current_phase == "SEARCH":
                if last_message.source == "SearchAgent":
                    return "Validator"
                if last_message.source == "Validator":
                    return "Critic"
                if last_message.source == "Critic":
                    if "REFINE_SEARCH" in content:
                        self.refine_search_count += 1

                        return "QueryPlanner"
                    if "PROCEED_TO_SYNTHESIS" in content:
                        self.current_phase = "SYNTHESIS"
                        self.proceed_to_synthesis_count += 1

                        # Store approved papers from critic for reference validation
                        try:
                            self.approved_papers = json.loads(last_message.content).get("approved_papers", [])
                        except:
                            self.approved_papers = []
                        return "SynthesisAgent"
                if last_message.source == "QueryPlanner":
                    return "SearchAgent"

            if self.current_phase == "SYNTHESIS":
                if last_message.source == "SynthesisAgent":
                    if "SYNTHESIS_COMPLETE" in content:
                        return "ReferenceConsistencyCritic"
                if last_message.source == "ReferenceConsistencyCritic":
                    if "REVISE_NEEDED" in content:
                        self.reference_validation_count += 1  # Increment counter


                        return "SynthesisAgent"
                    if "PROCEED_TO_FORMATTING" in content:
                        return "FormatterAgent"
                if last_message.source == "FormatterAgent" and "TERMINATE" in content:
                    return None

            return None

        # Initialize the team with all agents
        self.team = SelectorGroupChat(
            participants=[
                self.query_planner,
                self.search_agent,
                self.validator_agent,
                self.critic_agent,
                self.synthesis_agent,
                self.reference_consistency_critic,  # Add new critic
                self.formatter_agent
            ],
            model_client=self.model_client,
            termination_condition=TextMentionTermination("TERMINATE"),
            selector_func=selector_func
        )

        
    async def process_query(self, query: str):
        try:
            initial_message = TextMessage(
                content=f"""Research Topic: {query}

                Task Flow:
                1. Search & Initial Validation:
                   - Generate focused queries
                   - Execute PubMed searches
                   - Validate paper quality
                   - Critic evaluates search results
                   This phase repeats until Critic approves

                2. Synthesis & Validation Cycle:
                   a. Initial Synthesis:
                      - Write introduction using validated papers
                   
                   b. Validation Loop:
                      - Critic reviews synthesis quality and coverage
                      - If revision needed → back to synthesis
                      - If approved → proceed to reference check
                   
                   c. Reference Consistency:
                      - ReferenceConsistencyCritic validates all citations
                      - Verifies citations match approved papers exactly
                      - If citations invalid → back to synthesis
                      - If valid → proceed to formatting

                3. Final Formatting:
                   - Format validated synthesis and references
                   
                Requirements:
                - Only use critic-approved papers
                - All synthesis revisions require critic approval
                - Citations must exactly match approved references
                - No reference hallucination
                - Maintain academic integrity through validation cycles""",
                source="user"
            )
            
            async for message in self.team.run_stream(task=initial_message):
                if hasattr(message, 'source') and hasattr(message, 'content'):
                    yield {
                        "type": "update",
                        "agent": message.source,
                        "content": message.content
                    }
            
            await self.team.reset()
            
        except Exception as e:
            yield {
                "type": "error",
                "message": str(e)
            }
