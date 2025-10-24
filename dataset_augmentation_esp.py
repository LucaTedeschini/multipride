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
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'), 
    base_url="https://api.deepseek.com"
)

system_prompt = """Eres un anotador para una tarea de clasificación. En la entrada recibirás la biografía de un usuario de Twitter y algunos de sus tuits.  
Tu tarea es decidir si el usuario en cuestión forma parte o no de la comunidad LGBT.  
Basarás tu decisión en la forma de escribir, el contenido de la biografía y de los tuits, y en todos los factores que consideres relevantes.  
La entrada que te proporcionaré tendrá el formato TWEET - BIO.  
La salida que quiero es simplemente un número:  
0 si el usuario NO pertenece a la comunidad LGBT,  
1 si sí pertenece a la comunidad.  
Ejemplo de interacción:  
ENTRADA: "fuck gender rules and the rules of society || bts || exo" - "pansexual, genderqueer and polyamorous 🏳️‍🌈 || she/her || unito dams"  
SALIDA: 1"""
################

## Data Loading ##
console.print("[bold]Loading dataset...[/bold]")
ita = pd.read_csv("dataset/train_es.csv")
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
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            stream=False
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
ita.to_csv("augmented_es.csv", index=False)
console.print("[bold green]:white_check_mark: File saved as [cyan]augmented_es.csv[/cyan][/bold green]")
console.print(ita.head())