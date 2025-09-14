# Testes de redes neurais em C++
Meus projetos de testes redes neurais/vetorização em C++

1. ativações: degrau, sigmoid, tanh, ReLU, GELU, e mais, + deverivadas.
2. tokenizadores: tokenzador sub palavra.
3. utilitários: pesos, atualização de pesos, perda, erro, saída, métricas, matrizes 2D/3D + vetores.

## para testar:

```Cpp
#include <toke.h>
#include <util.h>

int main() {
  testeU(); // testa todos os utilitários
  testeT(); // testa o tokenizador
  return 0;
}
```

implementação feita do zero, sem bibliotecas de IA prontas.

## exemplos
você pode encontrar em exemplos:

1. percpetron.cpp
