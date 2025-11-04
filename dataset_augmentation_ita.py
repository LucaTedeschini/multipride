import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time
from rich.console import Console
from rich.progress import track

### Setup ###
console = Console()
load_dotenv()
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

system_prompt = """Sei un annotatore per un task di classificazione. In input riceverai la bio di un utente Twitter e alcuni suoi tweet.
Il tuo compito è decidere se l'utente in questione fa parte o meno della comunità LGBT.
Basa la tua scelta sul modo di scrivere, sul contenuto della bio e dei tweet, e su tutti i fattori che ritieni rilevanti.
L'input che ti fornirò sarà nel formato TWEET - BIO.
L'output che voglio è semplicemente un numero:
0 se l'utente NON appartiene alla comunità LGBT,
1 se invece appartiene alla comunità.
Esempio di interazione:
INPUT: "fuck gender rules and the rules of society || bts || exo" - "pansexual, genderqueer and polyamorous 🏳️‍🌈 || she/her || unito dams"
OUTPUT: 1"""
################

## Data Loading ##
console.print("[bold]Loading dataset...[/bold]")
ita = pd.read_csv("dataset/train_it.csv")
console.print("[bold green]Dataset loaded.[/bold green]")

## API Processing ##
lgbt = []
iterable = zip(ita["text"], ita["bio"])

for text, bio in track(iterable, description="[cyan]Processing entries...[/cyan]", total=len(ita)):

    bio = "" if str(bio) == "nan" else bio
    user_message = f'"{text}" - "{bio}"'

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            stream=False,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}\nRetrying in 2 minutes...[/bold red]")
        answer = "error"
        time.sleep(120)

    if answer in ["1", "0"]:
        answer = int(answer)
    else:
        # Mark ambiguous/error responses
        answer = 0.5

    lgbt.append(answer)
    time.sleep(0.5)

## Finalizing ##
ita["lgbt"] = lgbt
console.print("\n[bold yellow]Saving augmented dataset...[/bold yellow]")
ita.to_csv("augmented_it.csv", index=False)
console.print("[bold green]:white_check_mark: File saved as [cyan]augmented_it.csv[/cyan][/bold green]")
console.print(ita.head())
