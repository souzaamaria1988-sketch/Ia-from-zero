# Amostras geradas pelo GPT-Machado (1.353.728 parâmetros, do zero)

## v2 (após fase 2 de treino, val 2,79) — GGUF Q8_0, prompt "— Capitu, você me", seed 7

```text
no de uma vida, mas a água guerra Livna Humanilha. Abriu-a
aocasião, não exceberia com as suas intenções do negócio, como os seus
píneos. Tanto se manifesta de coro, que não passavam nada mais do
ponto. Naquele leitor que deve ter a sua idade delas e concorduir aos pés,
deram devez sós, com o padrinho de mim mesmo, é fugir um céu, uma das
barrezes, conjuge, é que está a minha c
```
*Comparado à v1 (mesmo prompt, mesma seed): períodos mais longos e estruturados,
concordâncias mais frequentes e vocabulário bem mais real.*

## REPL PyTorch (v2), prompt "— Naquela noite, o capitão das"

```text
— Naquela noite, o capitão das banças?
— Tinha diganh!
— Aclausa vala, agora me podia a irazar-mo que não pai.
```

## PyTorch (checkpoint final da fase 2, temp 0.85, top_k 64)

```text
edita. Pois seria ainda por revolunta e os inosentídos. Evia logo, mas
já, o testado tempo pior e tropésetas para sair, e disso, pois, para ir a consciência,
aposto as delicações do chão, aos acabaros, que se infinitivesse. Os só
defeitos coronelirom os vestões, que o homem mirar e acerte, na maneira
transformação de minórico, outro modio, qualquer que é circunstância com o
fundo. Uma pensaginação decluir no fim de outro. Deus haverível.
```

## v1 (fase 1, val 3,15) — PyTorch, temp 0.85, top_k 64

```text
 e palavras com a do Onde Brasileiros.
A casa não é esterente, porque o que forme estão só essas pelo de alguns pudes
blúnias.
No admendiante três se tão assim por dele, carrocandar-se o homem e
pósticular aberração de céux, nem a quem fambês fazer no criante
minha brata das levas, que o presceito, trabém, mas a ama do caso do dia
```

## v1 — GGUF Q8_0, prompt "— Capitu, você me", seed 7 (para comparar com a v2)

```text
com que os carros.
A escrevidavam passado por uma modenta de uma passa, e
diziam o cortado de amores, o meu só
de convisituiu o do que lhe sobralmente corrigem com ele. Ainda
ventitaria-se, ela rementir o áculo e sobrar com olho efordo,
```

## Curva de aprendizado (val loss)

| iteração | 0 | 200 | 400 | 600 | 1000 | 1400 | 1800 | 2400 | — | 2800 | 3200 | 3600 | 4000 | 4400 | **4800** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val loss | 6,257 | 3,918 | 3,790 | 3,746 | 3,547 | 3,351 | 3,231 | 3,151 | — | 3,149 | 3,040 | 2,939 | 2,851 | 2,831 | **2,789** |

Começa em ln(512) ≈ 6,24 (palpite aleatório uniforme) e desce monotonicamente.
A fase 2 (reinício de taxa de aprendizado em 6e-4, dados vistos em nova ordem)
quebrou o platô da fase 1 e rendeu mais −0,36 de loss.

