# PRK/TEK Simbolički Analizator Kola

Mali SPICE-like alat za simboličku analizu električnih kola i idealnih vodova.

## Šta alat radi

- Parsira netlist fajl (`.cir`).
- Formira i rešava MNA sistem u Laplasovom domenu.
- Računa simbolički:
  - napone čvorova,
  - prenosnu funkciju `H(s) = Vout(s)/Vin(s)`,
  - nule/polove,
  - `H(jω)` i amplitudski odziv `A(ω) = |H(jω)|`,
  - 3 dB karakteristike (simbolički ili numerički).
- Vremenski odziv:
  - simbolički (inverzna Laplasova, sa `Heaviside(t)`),
  - numerički fallback za brži plot.
- Podržava i GUI (web interfejs).

## Podržani elementi

- `R`, `L`, `C`, `V`, `E` (VCVS),
- `O` (idealan operacioni pojačavač),
- `T` (idealan vod sa parametrima `Zc` i `tau`).

## Struktura projekta

- `engine/spice_parser.py` - parser netliste
- `engine/mna.py` - formiranje MNA jednačina
- `engine/analysis.py` - simbolička/numerička analiza i plotovi
- `run.py` - CLI pokretanje
- `web/app.py` - lokalni web server/API
- `web/index.html` - GUI
- `examples/` - primeri netlista

## Instalacija i pokretanje

Preporuka je virtualno okruženje:

```bash
python3 -m venv venv
source venv/bin/activate
pip install sympy numpy matplotlib scipy
```

Ako je `scipy` nedostupan, deo numeričkih funkcija radi sa graceful fallback-om.

### CLI

Prikaži sve opcije:

```bash
venv/bin/python run.py --help
```

Osnovni primer:

```bash
venv/bin/python run.py --netlist examples/rc.cir --in-node in --out-node out --mode symbolic
```

Primer samo za prenosnu funkciju i frekvencijsku analizu:

```bash
venv/bin/python run.py --netlist examples/khn.cir --in-node 1 --out-node 3 --mode symbolic --tf-only --symbolic-freq --amp-plot --amp-w-max 3
```

Merenja (primer):

```bash
venv/bin/python run.py --netlist examples/tline_task12_cap.cir --measure "V(n2)" --measure "I(R1)" --measure-time --measure-plot "V(n2)"
```

## GUI

Pokretanje:

```bash
venv/bin/python web/app.py
```

Otvoriti:

```text
http://127.0.0.1:8000
```

GUI podržava:
- izbor netliste iz `examples/`,
- `Create Netlist` stranicu za Falstad paste, ručnu doradu i snimanje u `examples/`,
- prikaz kompaktnih LaTeX izraza,
- frekvencijsku analizu,
- amplitudsku karakteristiku,
- merenja (`V(node)`, `I(element)`) i njihove plotove,
- prikaz slike kola iz netliste preko komentara:
  - `* IMG: assets/circuit_images/neka_slika.png`

Napomena za Falstad import (trenutni MVP):
- podržani su `R`, `C`, `L`, `V`, `I`, žice, masa i Falstad `a` op-amp,
- Falstad `a` op-amp se prevodi u naš idealni `O` element,
- postoji i `To Symbolic` korak za brzu zamenu numeričkih vrednosti sa `R`, `C`, `L`, `Ug`, ...
- unsupported Falstad elementi trenutno vraćaju jasnu grešku umesto tihog pogrešnog prevoda.

## Bitna napomena za idealni OPAMP

Idealni op-amp je podržan direktno (element `O`) i kroz VCVS model (`E` sa velikim pojačanjem ili simboličkim parametrom), uz očuvanje idealne semantike.

## Kako okačiti projekat na GitHub

Ako već nemaš repozitorijum:

1. Napravi novi prazan repo na GitHub-u (npr. `prk-symbolic-analyzer`).
2. U root-u projekta pokreni:

```bash
git init
git add .
git commit -m "Initial commit: PRK/TEK symbolic analyzer"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

Ako repo već postoji lokalno:

```bash
git add .
git commit -m "Update: README (sr), GUI i analiza"
git push
```

## Tipični problemi

- `ModuleNotFoundError: No module named 'engine'`  
  Pokreni iz root foldera projekta:
  `venv/bin/python web/app.py`

- `FileNotFoundError` za netlist  
  Koristi tačnu putanju, npr. `examples/rc.cir`.

- GUI ne osvežava izmene  
  Restartuj server (`Ctrl+C`, pa ponovo pokretanje) i uradi hard refresh u browseru (`Ctrl+F5`).
