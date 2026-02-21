"""
CLI Interface for HR Query Assistant
Allows testing the system from command line without a web UI.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
console = Console()


def print_banner():
    console.print(Panel.fit(
    "[bold #6366f1]🏢 Enterprise HR Query Assistant[/bold #6366f1]\n"
    "[dim]Powered by LangGraph + Groq AI[/dim]",
    border_style="#6366f1",
    padding=(1, 4),
))


def print_response(result: dict):
    # Category badge color
    category_colors = {
        "leave_policy": "green",
        "reimbursement": "blue",
        "insurance": "cyan",
        "payroll": "yellow",
        "onboarding": "magenta",
        "benefits": "green",
        "remote_work": "blue",
        "general_policy": "white",
        "unknown": "red",
    }
    cat_color = category_colors.get(result.get("category", "unknown"), "white")

    # Status line
    confidence = result.get("confidence", 0) * 100
    escalated = result.get("escalated", False)
    conf_color = "green" if confidence >= 70 else "yellow" if confidence >= 50 else "red"

    console.print()
    console.print(f"  [dim]Category:[/dim] [{cat_color}]{result.get('category', 'unknown')}[/{cat_color}]  "
                  f"[dim]Confidence:[/dim] [{conf_color}]{confidence:.0f}%[/{conf_color}]  "
                  f"[dim]Time:[/dim] {result.get('response_time_ms', 0)}ms  "
                  f"{'[bold red]⚠ ESCALATED[/bold red]' if escalated else '[bold green]✓ ANSWERED[/bold green]'}")

    # Response panel
    style = "red" if escalated else "indigo"
    title = "HR Response (Escalated to HR Team)" if escalated else "HR Response"
    console.print(Panel(
        result.get("response", "No response generated"),
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2),
    ))

    # Sources
    sources = result.get("sources", [])
    if sources:
        console.print(f"  [dim]📄 Sources: {', '.join(sources)}[/dim]")

    console.print()


async def interactive_mode(employee_id: str, department: str, role: str):
    """Interactive chat mode for HR queries."""
    # from src.graphs.hr_query_graph import process_hr_query
    from hr_query_graph import process_hr_query

    console.print(f"\n[bold]Employee:[/bold] {employee_id} | [bold]Dept:[/bold] {department} | [bold]Role:[/bold] {role}")
    console.print("[dim]Type your HR question below. Type 'quit' to exit.\n[/dim]")

    while True:
        try:
            # query = Prompt.ask("\n[bold indigo]Your Question[/bold indigo]")
            query = Prompt.ask("\n[bold #6366f1]Your Question[/bold #6366f1]")
            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]Goodbye! 👋[/dim]\n")
                break

            if not query.strip():
                continue

            with console.status("[bold indigo]Processing your query...[/bold indigo]", spinner="dots"):
                result = await process_hr_query(
                    query=query,
                    employee_id=employee_id,
                    department=department,
                    role=role,
                )

            print_response(result)

            # Feedback
            satisfied = Confirm.ask("[dim]Was this response helpful?[/dim]", default=True)
            if not satisfied:
                console.print("[dim]Thank you for your feedback. The HR team will improve this response.[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[dim]Session ended. Goodbye![/dim]\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")


async def demo_mode():
    """Run demo queries to showcase the system."""
    # from src.graphs.hr_query_graph import process_hr_query
    from hr_query_graph import process_hr_query

    demo_queries = [
        {
            "query": "How many casual leaves am I entitled to per year?",
            "employee_id": "EMP001",
            "department": "Engineering",
            "role": "employee",
        },
        {
            "query": "What is the process to claim medical reimbursement?",
            "employee_id": "EMP002",
            "department": "Sales",
            "role": "employee",
        },
        {
            "query": "I want to raise a complaint against my manager for unfair treatment.",
            "employee_id": "EMP003",
            "department": "Marketing",
            "role": "employee",
        },
        {
            "query": "Can I work from home 3 days a week?",
            "employee_id": "EMP004",
            "department": "Finance",
            "role": "manager",
        },
    ]

    console.print("\n[bold]🎬 Running Demo Queries...[/bold]\n")

    for i, demo in enumerate(demo_queries, 1):
        console.print(f"[bold dim]Demo Query {i}/{len(demo_queries)}:[/bold dim]")
        console.print(f"[italic]'{demo['query']}'[/italic]")

        with console.status("Processing...", spinner="dots"):
            result = await process_hr_query(**demo)

        print_response(result)

        if i < len(demo_queries):
            input("Press Enter for next demo...")


def ingest_documents_cmd():
    """Ingest all documents from the documents directory."""
    # from src.tools.document_ingestion import get_ingestion_tool
    from document_ingestion import get_ingestion_tool

    console.print("\n[bold]📂 Ingesting HR Documents...[/bold]\n")
    tool = get_ingestion_tool()
    results = tool.ingest_directory()

    table = Table(title="Document Ingestion Results", show_header=True)
    table.add_column("Filename", style="bold")
    table.add_column("Category")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    for r in results:
        status = "✅ Success" if r.get("status") == "success" else "❌ Failed"
        table.add_row(
            r.get("filename", "N/A"),
            r.get("category", "N/A"),
            str(r.get("chunks", 0)),
            status,
        )

    if results:
        console.print(table)
    else:
        console.print("[yellow]No documents found. Add PDF/DOCX/TXT files to data/documents/[/yellow]")

    # Show stats
    stats = tool.get_collection_stats()
    console.print(f"\n[bold green]Vector Store: {stats['total_chunks']} chunks indexed[/bold green]\n")


def show_help():
    # table = Table(show_header=True, header_style="bold indigo")
    table = Table(show_header=True, header_style="bold #6366f1")
    table.add_column("Command")
    table.add_column("Description")

    commands = [
        ("chat", "Start interactive HR query session"),
        ("demo", "Run demo queries to test the system"),
        ("ingest", "Ingest documents from data/documents/"),
        ("server", "Start the FastAPI backend server"),
        ("dashboard", "Start the HR Analytics Dashboard"),
        ("help", "Show this help message"),
    ]

    for cmd, desc in commands:
        table.add_row(f"[bold]{cmd}[/bold]", desc)

    console.print("\n[bold]Available Commands:[/bold]")
    console.print(table)
    console.print()


async def main():
    print_banner()

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "chat":
        employee_id = Prompt.ask("Employee ID", default="EMP001")
        department = Prompt.ask("Department", default="Engineering")
        role = Prompt.ask("Role", choices=["employee", "manager", "hr_admin", "hr_manager", "executive"], default="employee")
        await interactive_mode(employee_id, department, role)

    elif cmd == "demo":
        await demo_mode()

    elif cmd == "ingest":
        ingest_documents_cmd()

    elif cmd == "server":
        console.print(f"\n[bold]🚀 Starting FastAPI Server on port {os.getenv('API_PORT', 8000)}...[/bold]\n")
        import uvicorn
        uvicorn.run("src.api.server:app", host="0.0.0.0", port=int(os.getenv("API_PORT", 8000)), reload=True)

    elif cmd == "dashboard":
        console.print(f"\n[bold]📊 Starting HR Analytics Dashboard on port {os.getenv('DASHBOARD_PORT', 8050)}...[/bold]\n")
        import subprocess
        subprocess.run([sys.executable, "src/dashboard/analytics_dashboard.py"])

    else:
        show_help()


if __name__ == "__main__":
    asyncio.run(main())
