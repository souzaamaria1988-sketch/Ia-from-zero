# Amostras geradas pelo GPT-Machado (1.353.728 parâmetros, do zero)

## PyTorch (checkpoint final, temp 0.85, top_k 64)

```text
 e palavras com a do Onde Brasileiros.
A casa não é esterente, porque o que forme estão só essas pelo de alguns pudes
blúnias.
No admendiante três se tão assim por dele, carrocandar-se o homem e
pósticular aberração de céux, nem a quem fambês fazer no criante
minha brata das levas, que o presceito, trabém, mas a ama do caso do dia
escom os olhos que certam na cama. A má
quilogou como o pagar, chegou Moquilia a voz a logista da Rua),
paradando à moncerta, a rua, que, dessabrita do bondente João Isaba.
Não conselho e naquele dia segundo.
— Posso mesmo torna-me, instido esces, não quer, vai a
meição.
```

## GGUF Q8_0 (1,58 MB) — llama.cpp, prompt "— Capitu, você me", seed 7

```text
com que os carros.
A escrevidavam passado por uma modenta de uma passa, e
diziam o cortado de amores, o meu só
de convisituiu o do que lhe sobralmente corrigem com ele. Ainda
ventitaria-se, ela rementir o áculo e sobrar com olho efordo,
pressou antes, não disse alguém depois com ondente. Dide aquela
possão, e boma de uma séria, os amigos.
A falar aba do meu homem não tudo, quando nos espagar
```
*Velocidade: ~980 tokens/s em 2 núcleos de CPU.*

## GGUF Q4_K_M (1,13 MB) — mesmo prompt, mesmo seed

```text
combrando, e tudo se false da bonesada. De a três
provimentos de duas, se tudo irturuas de bancaria. Destou as
remais, mas eram-lhe abistação. Aidmos negreu-lhe a passar
neralistas. Vos aí também, para onde seguiu na
presição de João. Já que foi assim título.
Como lá mais simples, abostravam-lhe o que realou-lhe depois
conhecimento depois. Flora que sou, por extrar a s
```
*Velocidade: ~1.089 tokens/s em 2 núcleos de CPU.*

## Curva de aprendizado (val loss)

| iteração | 0 | 200 | 400 | 600 | 1000 | 1400 | 1800 | 2400 |
|---|---|---|---|---|---|---|---|---|
| val loss | 6,257 | 3,918 | 3,790 | 3,746 | 3,547 | 3,351 | 3,231 | **3,151** |

Começa em ln(512) ≈ 6,24 (palpite aleatório uniforme) e desce monotonicamente.
Para 1,35M de parâmetros em 42 min de 2 CPUs, é exatamente o esperado: o modelo
aprendeu ortografia, vocabulário e ritmo da prosa de Machado; "sabedoria" de
conteúdo exigiria ordens de magnitude mais parâmetros/dados.
