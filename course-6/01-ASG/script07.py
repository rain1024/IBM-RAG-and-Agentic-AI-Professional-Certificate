"""
Facebook Posts Agent Demo in LangChain
================================================================================

Module: script07.py
Author: @rain1024
Version: 1.0.0
Last Modified: 2025
Development Environment: Cursor IDE with Claude-4-Sonnet

DESCRIPTION:
    Demo tool showing how agents can interact with real-world data and perform tasks.
    This Facebook Posts Agent can write posts and find posts for users using structured tools.
    Demonstrates practical AI agent capabilities for social media management.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=1000
)

# Mock database for demo purposes
MOCK_POSTS_DB = [
    {
        "id": "post_001",
        "user": "john_doe",
        "content": "Just finished my morning workout! 💪 #fitness #motivation",
        "timestamp": "2024-01-15T08:30:00Z",
        "visibility": "public",
        "likes": 25,
        "comments": 3,
        "tags": ["fitness", "motivation"]
    },
    {
        "id": "post_002", 
        "user": "jane_smith",
        "content": "Beautiful sunset at the beach today 🌅 #nature #photography",
        "timestamp": "2024-01-14T19:45:00Z",
        "visibility": "public",
        "likes": 42,
        "comments": 8,
        "tags": ["nature", "photography"]
    },
    {
        "id": "post_003",
        "user": "john_doe",
        "content": "Working on a new project. Excited to share more soon! #tech #innovation",
        "timestamp": "2024-01-13T14:20:00Z",
        "visibility": "friends",
        "likes": 15,
        "comments": 5,
        "tags": ["tech", "innovation"]
    }
]

# Define input schemas for structured tools
class WritePostInput(BaseModel):
    """Input schema for writing Facebook posts"""
    content: str = Field(description="Content of the post to write")
    visibility: Literal["public", "friends", "private"] = Field(
        default="public",
        description="Post visibility setting"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="List of hashtags for the post (without # symbol)"
    )
    user: str = Field(description="Username of the person posting")

class FindPostsInput(BaseModel):
    """Input schema for finding Facebook posts"""
    user: Optional[str] = Field(
        default=None,
        description="Username to find posts for (optional)"
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords to search for in post content"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tags to filter posts by"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of posts to return"
    )

# Define structured tools for Facebook posts
@tool(args_schema=WritePostInput)
def write_facebook_post(content: str, user: str, visibility: str = "public", tags: Optional[List[str]] = None) -> str:
    """Write a new Facebook post with structured input validation."""
    try:
        # Generate a new post ID
        post_id = f"post_{len(MOCK_POSTS_DB) + 1:03d}"
        
        # Create new post
        new_post = {
            "id": post_id,
            "user": user,
            "content": content,
            "timestamp": datetime.now().isoformat() + "Z",
            "visibility": visibility,
            "likes": 0,
            "comments": 0,
            "tags": tags or []
        }
        
        # Add to mock database
        MOCK_POSTS_DB.append(new_post)
        
        # Format response
        tag_str = " ".join([f"#{tag}" for tag in (tags or [])])
        full_content = f"{content} {tag_str}".strip()
        
        return f"✅ Post created successfully!\n" \
               f"📝 User: {user}\n" \
               f"📄 Content: {full_content}\n" \
               f"🔒 Visibility: {visibility}\n" \
               f"🆔 Post ID: {post_id}"
               
    except Exception as e:
        return f"❌ Error creating post: {e}"

@tool(args_schema=FindPostsInput)
def find_facebook_posts(user: Optional[str] = None, keywords: Optional[List[str]] = None, 
                       tags: Optional[List[str]] = None, limit: int = 5) -> str:
    """Find Facebook posts with structured search criteria."""
    try:
        filtered_posts = MOCK_POSTS_DB.copy()
        
        # Filter by user
        if user:
            filtered_posts = [post for post in filtered_posts if post["user"].lower() == user.lower()]
        
        # Filter by keywords in content
        if keywords:
            filtered_posts = [
                post for post in filtered_posts 
                if any(keyword.lower() in post["content"].lower() for keyword in keywords)
            ]
        
        # Filter by tags
        if tags:
            filtered_posts = [
                post for post in filtered_posts 
                if any(tag.lower() in [t.lower() for t in post["tags"]] for tag in tags)
            ]
        
        # Limit results
        filtered_posts = filtered_posts[:limit]
        
        if not filtered_posts:
            return "🔍 No posts found matching your criteria."
        
        # Format results
        results = [f"📊 Found {len(filtered_posts)} post(s):\n"]
        
        for i, post in enumerate(filtered_posts, 1):
            results.append(
                f"{i}. 👤 {post['user']} ({post['timestamp'][:10]})\n"
                f"   📝 {post['content']}\n"
                f"   💬 {post['likes']} likes, {post['comments']} comments\n"
                f"   🔒 {post['visibility']}\n"
            )
        
        return "\n".join(results)
        
    except Exception as e:
        return f"❌ Error finding posts: {e}"

def create_facebook_posts_agent():
    """Create Facebook posts agent with structured tools"""
    
    # List of structured tools
    tools = [write_facebook_post, find_facebook_posts]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful Facebook Posts Agent with access to structured tools:

🔧 Available Tools:
- write_facebook_post: Create new Facebook posts with content, visibility settings, and tags
- find_facebook_posts: Search for existing posts by user, keywords, or tags

💡 Capabilities:
- Help users create engaging social media content
- Find relevant posts based on search criteria
- Manage post visibility and tagging
- Provide social media insights and suggestions

📝 When writing posts:
- Keep content engaging and appropriate
- Suggest relevant hashtags when applicable
- Consider the target audience and visibility setting

🔍 When finding posts:
- Use specific search criteria provided by the user
- Present results in a clear, organized format
- Provide relevant post metrics and details

Always use the appropriate tool based on the user's request."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def run_structured_tool_demo():
    """Run Facebook posts agent demonstration"""
    print("\n" + "="*60)
    print("FACEBOOK POSTS AGENT DEMO - REAL-WORLD DATA INTERACTION")
    print("="*60)
    
    # Create agent
    agent = create_facebook_posts_agent()
    
    # Demo queries showing real-world interaction capabilities
    demo_queries = [
        "Write a post for user 'alice_tech' about completing a Python course with tech and learning tags",
        "Find all posts by john_doe",
        "Search for posts containing the word 'fitness'",
        "Create a public post for 'bob_travel' about visiting Paris with travel and photography tags",
        "Find posts with the tag 'motivation'",
        "Write a friends-only post for 'sara_cook' about trying a new recipe",
        "Search for posts about 'sunset' or 'nature'"
    ]
    
    for query in demo_queries:
        print(f"\n{'='*50}")
        print(f"User: {query}")
        print(f"{'='*50}")
        
        try:
            # Get response from agent
            response = agent.invoke({"input": query})
            print(f"Agent: {response['output']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "-"*50)
    
    print("\n🎉 Facebook Posts Agent Demo completed!")
    print("💡 This demo shows how AI agents can interact with real-world data")
    print("   and perform practical tasks like social media management!")

if __name__ == "__main__":
    run_structured_tool_demo() 