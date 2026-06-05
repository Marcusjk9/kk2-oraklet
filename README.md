# KK2–Oraklet

Ladda upp ett CSV, ställ frågor om det, få svar från en lokal AI-modell.

## Kom igång

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Första gången laddas modellen ner automatiskt. Sedan är den cachad lokalt.

Swagger-UI finns på `http://localhost:8000/docs` om man föredrar att testa via webbläsaren.

## Använd API:t

Ladda upp ett dataset:

```bash
curl -X POST http://localhost:8000/data/upload -F "fil=@fifa_players.csv"
```

Ställ en fråga:

```bash
curl -X POST http://localhost:8000/ai/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Vilken spelare har högst overall_rating?"}'
```

Hämta statistik:

```bash
curl http://localhost:8000/data/stats
```

## Tester

```bash
uv run pytest app/tests/ -v
```

## Antaganden

- Modellen körs lokalt, ingen API-nyckel behövs.
- Datasetet sparas i minnet och försvinner när servern stängs av.
- Modellen ser bara numerisk statistik (min, max, medel osv.) – inte enskilda rader eller spelarnamn.
