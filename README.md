# Testes de redes neurais em C++
Meus projetos de testes redes neurais/vetorização em C++

1. ativações: degrau, sigmoid, tanh, ReLU, GELU, etc, e deverivadas.
2. tokenizadores: tokenzador sub palavra.
3. utilitários: pesos, atualização de pesos, perda, erro, saída, métricas.

## para testar:

```Cpp
#include <ativas.h>
#include <toke.h>
#include <util.h>

int main() {
  testeU(); // testa todos os utilitários
  testeT(); // testa o tokenizador
  return 0;
}
```

implementação feita do zero, sem bibliotecas de IA prontas.
