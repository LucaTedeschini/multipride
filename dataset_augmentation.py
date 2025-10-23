import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

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


ita = pd.read_csv("dataset/train_es.csv")
lgbt = []
c = 0

for text, bio in zip(ita["text"], ita["bio"]):
    c += 1
    print(f"{c} / {len(ita)}")
    
    if str(bio) == "nan":
        bio = ""
    
    user_message = f'"{text}" - "{bio}"'
    print(user_message)
    
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
        print(f"Si è verificato un errore: {e}\n\nRiprovo tra 2 minuti...")
        answer = "error"
        time.sleep(120)
    
    if answer == "1" or answer == "0":
        answer = int(answer)
    else:
        answer = 0.5
    
    lgbt.append(answer)
    print(answer)
    print("--" * 20)
    time.sleep(0.5)

ita["lgbt"] = lgbt
print(ita.head())
ita.to_csv("augmented_es.csv", index=False)