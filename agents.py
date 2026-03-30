from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
    max_tokens=2000
)

# ============================================================
# AGENTS
# ============================================================

research_agent = Agent(
    role="Product Research Specialist",
    goal="Find and evaluate dropshipping products with high demand and profit potential",
    backstory="""You are an expert e-commerce product researcher with 10 years of experience
    identifying winning dropshipping products. You understand market trends, consumer psychology,
    and what makes products go viral on TikTok and Instagram. You specialize in AliExpress
    products that can be sold on Shopify stores targeting US, UK, and Kenyan markets.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

scoring_agent = Agent(
    role="Product Scoring Analyst",
    goal="Score products on demand, margin, competition, and trend potential",
    backstory="""You are a data-driven analyst who evaluates e-commerce products using
    quantitative scoring frameworks. You assign scores from 0-100 for demand, trend,
    and calculate realistic margin percentages. You always output structured data
    including a final GO, CAUTION, or SKIP verdict.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

content_agent = Agent(
    role="E-commerce Content Creator",
    goal="Create platform-specific content packages that drive sales",
    backstory="""You are a social media and e-commerce copywriting expert who creates
    high-converting content for dropshipping stores. You write TikTok hooks, Instagram
    captions, Pinterest descriptions, Facebook posts, email subject lines, and SEO-optimized
    Shopify product descriptions. Your content is always fresh, engaging, and tailored
    to the platform.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

niche_agent = Agent(
    role="Niche Market Analyst",
    goal="Analyze entire product niches for profitability and market opportunity",
    backstory="""You are a market research specialist who evaluates entire product
    categories for dropshipping viability. You identify the top 5 winning products
    in any niche, assess competition levels, and recommend entry strategies.
    You have deep knowledge of consumer trends in beauty, fitness, home, tech,
    and lifestyle categories.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

# ============================================================
# TASK BUILDERS (now accept memory_context)
# ============================================================

def build_research_task(product_input, memory_context=""):
    return Task(
        description=f"""Research this product for dropshipping viability: {product_input}

        {memory_context}

        Provide a comprehensive analysis including:
        1. Product overview and target audience
        2. Demand analysis - who is buying this and why
        3. AliExpress price range (estimate) and recommended selling price
        4. Estimated profit margin percentage
        5. Competition level (Low/Medium/High) with reasoning
        6. Trend direction - is this growing or declining
        7. Risk flags - quality issues, shipping problems, saturation
        8. Top 3 next steps if pursuing this product
        9. Final verdict: GO / CAUTION / SKIP with one-line reason

        End your response with this exact line:
        SCORES:{{"demand":XX,"margin":XX,"competition":"LOW/MED/HIGH","trend":XX,"verdict":"GO/CAUTION/SKIP"}}
        """,
        agent=research_agent,
        expected_output="Detailed product research report with scores"
    )

def build_scoring_task(product_input, memory_context=""):
    return Task(
        description=f"""Score this product for dropshipping: {product_input}

        {memory_context}

        Provide numerical scores and analysis:
        - Demand Score (0-100): based on search trends and social media interest
        - Margin Score (0-100): based on typical AliExpress vs market price
        - Competition Score (0-100): lower score = more competition
        - Trend Score (0-100): based on growth trajectory
        - Overall Verdict: GO (score 70+), CAUTION (40-69), SKIP (below 40)

        End your response with this exact line:
        SCORES:{{"demand":XX,"margin":XX,"competition":"LOW/MED/HIGH","trend":XX,"verdict":"GO/CAUTION/SKIP"}}
        """,
        agent=scoring_agent,
        expected_output="Scored product analysis with verdict"
    )

def build_niche_task(niche_input, memory_context=""):
    return Task(
        description=f"""Analyze this product niche for dropshipping: {niche_input}

        {memory_context}

        Provide:
        1. Niche overview - market size, buyer demographics, average order value
        2. Top 5 winning products in this niche with price ranges and margins
        3. Overall niche metrics - demand, competition, entry difficulty
        4. Best content platforms for this niche
        5. Common pitfalls to avoid
        6. SmartHaven verdict - should we pursue this niche?

        End your response with this exact line:
        SCORES:{{"demand":XX,"margin":XX,"competition":"LOW/MED/HIGH","trend":XX,"verdict":"GO/CAUTION/SKIP"}}
        """,
        agent=niche_agent,
        expected_output="Complete niche analysis report with scores"
    )

def build_content_task(product_input, memory_context=""):
    return Task(
        description=f"""Create a complete content package for this product: {product_input}

        {memory_context}

        Generate:
        1. 3 TikTok captions (hook-first, punchy, with hashtags)
        2. Instagram caption (lifestyle tone, 150-200 chars, hashtags)
        3. Pinterest description (keyword-rich, 200 chars)
        4. Facebook post (conversational, 2 short paragraphs)
        5. Email subject line (max 50 chars) + preview text (max 90 chars)
        6. Canva AI image prompt (detailed visual description)
        7. 5 SEO long-tail keywords for Shopify
        8. Shopify product description (150 words, SEO optimized)

        End your response with this exact line:
        SCORES:{{"demand":70,"margin":55,"competition":"MED","trend":65,"verdict":"GO"}}
        """,
        agent=content_agent,
        expected_output="Complete content package for all platforms"
    )

def build_comparison_task(products_input, memory_context=""):
    return Task(
        description=f"""Compare these products for dropshipping potential: {products_input}

        {memory_context}

        For each product provide:
        - Demand score (0-100)
        - Estimated margin percentage
        - Competition level
        - Trend direction
        - AliExpress price range / recommended selling price
        - Best target audience
        - Main risk factor

        Then provide:
        - Head-to-head winner for: best margin, easiest to market, lowest risk, highest potential
        - Final recommendation: which to pursue first, second, third

        End your response with this exact line:
        SCORES:{{"demand":75,"margin":60,"competition":"MED","trend":72,"verdict":"GO"}}
        """,
        agent=scoring_agent,
        expected_output="Product comparison report with rankings"
    )

# ============================================================
# CREW RUNNERS (now accept memory_context)
# ============================================================

def run_product_research(product_input: str, memory_context: str = "") -> str:
    task = build_research_task(product_input, memory_context)
    crew = Crew(
        agents=[research_agent, scoring_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return str(result)

def run_niche_analysis(niche_input: str, memory_context: str = "") -> str:
    task = build_niche_task(niche_input, memory_context)
    crew = Crew(
        agents=[niche_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return str(result)

def run_product_comparison(products_input: str, memory_context: str = "") -> str:
    task = build_comparison_task(products_input, memory_context)
    crew = Crew(
        agents=[scoring_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return str(result)

def run_content_brief(product_input: str, memory_context: str = "") -> str:
    task = build_content_task(product_input, memory_context)
    crew = Crew(
        agents=[content_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return str(result)
