# Pregled razvoja i ideja

Ovaj dokument ukratko objašnjava kako se projekat razvijao, koje su bile ključne ideje i kako su komponente povezane.

## 1. Polazna ideja i cilj

Cilj je bio napraviti mali, SPICE‑like alat koji ne radi samo numeriku već i simboliku: da korisnik dobije zatvorene formule za napone, prenosnu funkciju i ponašanje u vremenu/frekvenciji. Dodatni cilj je bio GUI kako bi alat bio upotrebljiv i bez komandne linije.

## 2. Parser netliste (ulaz)

Prvi korak je bio da se definira jednostavan format ulaza (netlista):

- svaka linija opisuje jedan element (R, L, C, V, E, O, T),
- čvorovi su označeni imenima,
- vrednosti mogu biti simboličke ili numeričke.

Parser prebacuje taj tekst u unutrašnju strukturu (elementi, čvorovi, parametri), koja je pogodna za matematičku obradu.

## 3. Model kola: MNA u kratkim crtama

Umesto da se pišu jednačine “ručno”, koristi se **Modified Nodal Analysis (MNA)**:

- Osnovna ideja: nepoznate su naponi čvorova (i dodatne struje za izvore napona i idealne elemente).
- Za svaki čvor piše se jednačina KCL (suma struja = 0).
- Elementi (R, L, C, zavisni izvori, idealni op‑amp, idealni vod) se “ubacuju” u matricu preko svojih relacija.

Rezultat je matrica:

```
A(s) * x(s) = z(s)
```

gde je:

- `x(s)` vektor nepoznatih (naponi čvorova + dodatne struje),
- `A(s)` matrica koja sadrži simboličke izraze u `s`,
- `z(s)` vektor pobude (izvori).

Ovim dobijamo sistem koji se **simbolički** rešava (npr. pomoću SymPy‑ja), pa dobijamo izraze kao funkcije od `s`.

## 4. Simbolička analiza

Na osnovu rešenja:

- računaju se naponi čvorova,
- prenosna funkcija `H(s) = Vout/Vin`,
- nule i polovi,
- frekvencijski odziv `H(jω)` i amplituda `A(ω)`.

Za 3 dB tačke moguće je:

- simboličko rešavanje (ako je izvodljivo),
- numeričko, kao fallback (brže i stabilnije).

## 5. Vremenski odziv

Za vremenski odziv koristi se:

- inverzna Laplasova transformacija za simbolički rezultat,
- numerička simulacija kao fallback kada simbolika postane teška.

Ideja je da korisnik dobije **tačnu formulu** kada je moguće, ali da uvek postoji praktična numerička alternativa.

## 6. GUI i upotrebljivost

Uz CLI je dodat web GUI:

- izbor netliste iz `examples/`,
- prikaz simboličkih izraza i LaTeX,
- grafici za frekvenciju i amplitude,
- merenja (V(node), I(element)).

Time se projekat pomera od “samo za skriptu” do alata koji može da koristi i neko ko nije naviknut na terminal.

## 7. Povezivanje backenda i frontenda

Ovo je rešeno jednostavno: backend server izlaže API, a frontend komunicira sa njim.

`Chat` (kratak opis povezivanja): Frontend šalje zahteve backendu (npr. “analiziraj ovu netlistu”), backend vraća rezultate (izrazi, vrednosti, grafici), frontend ih prikazuje korisniku.

## 8. Ključne ideje i motivacija

- **Simbolika pre numerike**: cilj nije samo “broj”, već formula.
- **Jednostavan ulaz, moćan izlaz**: netlista je kratka, rezultat je detaljan.
- **Fallback strategija**: simbolika kada može, numerika kada mora.
- **GUI kao most**: da alat bude koristan i van komandne linije.

## 9. Zaključak

Projekat je evoluirao od ideje “SPICE sa simbolikom” do funkcionalnog alata koji:

- prima netlistu,
- konstruiše MNA sistem,
- rešava ga simbolički,
- nudi i numeričke i grafičke rezultate,
- ima GUI za lakšu upotrebu.

