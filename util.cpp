#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <functional>

// funções de regularização:
std::vector<std::vector<float>> regularL1(const std::vector<std::vector<float>>& pesos, float lambda) {
    std::vector<std::vector<float>> res;
    for(const auto& linha : pesos) {
        std::vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * (p > 0 ? 1.0f : (p < 0 ? -1.0f : 0.0f)));
        res.push_back(novaLinha);
    }
    return res;
}

std::vector<std::vector<float>> regularL2(const std::vector<std::vector<float>>& pesos, float lambda) {
    std::vector<std::vector<float>> res;
    for(const auto& linha : pesos) {
        std::vector<float> novaLinha;
        for(float p : linha) novaLinha.push_back(lambda * p);
        res.push_back(novaLinha);
    }
    return res;
}

std::vector<std::vector<float>> dropout(std::vector<std::vector<float>> tensor, float taxa) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for(auto& linha : tensor) {
        for(float& valor : linha) {
            if(dis(gen) < taxa) valor = 0.0f;
            else valor /= (1.0f - taxa);
        }
    }
    return tensor;
}

std::vector<float> normEntrada(const std::vector<float>& vetor) {
    float max = *std::max_element(vetor.begin(), vetor.end());
    float min = *std::min_element(vetor.begin(), vetor.end());
    float amplitude = (max - min) > 1e-8f ? (max - min) : 1e-8f;
    
    std::vector<float> res;
    for(float x : vetor) res.push_back((x - min) / amplitude);
    return res;
}

std::vector<float> normZPonto(const std::vector<float>& v) {
    float soma = 0.0f;
    for(float x : v) soma += x;
    float media = soma / v.size();
    
    float variancia = 0.0f;
    for (float x : v) variancia += std::pow(x - media, 2);
    variancia /= v.size();
    
    float desvio = std::sqrt(variancia + 1e-8f);
    
    std::vector<float> res;
    for(float x : v) res.push_back((x - media) / desvio);
    return res;
}

// funções de metricas
float acuracia(const std::vector<std::vector<float>>& saida, const std::vector<std::vector<float>>& esperado) {
    int corretos = 0;
    for(size_t i = 0; i < saida.size(); i++) {
        int pred = std::distance(saida[i].begin(), std::max_element(saida[i].begin(), saida[i].end()));
        int real = std::distance(esperado[i].begin(), std::max_element(esperado[i].begin(), esperado[i].end()));
        if(pred == real) corretos++;
    }
    return static_cast<float>(corretos) / saida.size();
}

float precisao(const std::vector<std::vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fp = 0;
    for (size_t i = 1; i < confusao[0].size(); i++) {
        fp += confusao[0][i];
    }
    return static_cast<float>(tp) / (tp + fp + 1e-8f);
}

float recall(const std::vector<std::vector<int>>& confusao) {
    int tp = confusao[0][0];
    int fn = 0;
    for(size_t i = 1; i < confusao.size(); i++) fn += confusao[i][0];
    return static_cast<float>(tp) / (tp + fn + 1e-8f);
}

float f1Ponto(const std::vector<std::vector<int>>& confusao) {
    float p = precisao(confusao);
    float r = recall(confusao);
    return 2.0f * (p * r) / (p + r + 1e-8f);
}

float mse(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += std::pow(saida[i] - esperado[i], 2);
    return soma / saida.size();
}

float klDivergencia(const std::vector<float>& p, const std::vector<float>& q) {
    float soma = 0.0f;
    for(size_t i = 0; i < p.size(); i++) soma += p[i] * std::log((p[i] + 1e-12f) / (q[i] + 1e-12f));
    return soma;
}

float rocAuc(const std::vector<float>& pontos, const std::vector<int>& rotulos) {
    // cria pares [pontuação, rótulo] e ordenar por pontuação (decrescente)
    std::vector<std::pair<float, int>> pares;
    for(size_t i = 0; i < pontos.size(); i++) pares.push_back({pontos[i], rotulos[i]});
    
    std::sort(pares.begin(), pares.end(), 
        [](const auto& a, const auto& b) {
            return a.first > b.first;
    });
    
    float auc = 0.0f;
    int fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
    
    for(const auto& par : pares) {
        if(par.second == 1) tp++;
        else fp++;
        
        auc += (fp - fpPrev) * (tp + tpPrev) / 2.0f;
        fpPrev = fp;
        tpPrev = tp;
    }
    return auc / (tp * fp);
}

// funções de erro:
float erroAbsolutoMedio(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) soma += std::abs(saida[i] - esperado[i]);
    return soma / saida.size();
}

float erroQuadradoEsperado(const std::vector<float>& saida, const std::vector<float>& esperado) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        soma += 0.5f * diff * diff;
    }
    return soma;
}

std::vector<float> derivadaErro(const std::vector<float>& saida, const std::vector<float>& esperado) {
    std::vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) deriv[i] = saida[i] - esperado[i];
    return deriv;
}

float entropiaCruzada(const std::vector<float>& y, const std::vector<float>& yChapeu) {
    float soma = 0.0f;
    for(size_t i = 0; i < y.size(); i++) soma += y[i] * std::log(yChapeu[i] + 1e-12f);
    return -soma;
}

std::vector<float> derivadaEntropiaCruzada(const std::vector<float>& y, const std::vector<float>& yChapeu) {
    std::vector<float> deriv(yChapeu.size());
    for(size_t i = 0; i < yChapeu.size(); i++) deriv[i] = yChapeu[i] - y[i];
    return deriv;
}

float huberPerda(const std::vector<float>& saida, const std::vector<float>& esperado, float delta = 1.0f) {
    float soma = 0.0f;
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        if(std::abs(diff) <= delta) soma += 0.5f * diff * diff;
        else soma += delta * (std::abs(diff) - 0.5f * delta);
    }
    return soma / saida.size();
}

std::vector<float> derivadaHuber(const std::vector<float>& saida, const std::vector<float>& esperado, float delta = 1.0f) {
    std::vector<float> deriv(saida.size());
    for(size_t i = 0; i < saida.size(); i++) {
        float diff = saida[i] - esperado[i];
        
        if(std::abs(diff) <= delta) deriv[i] = diff;
        else deriv[i] = delta * (diff > 0 ? 1.0f : -1.0f);
    }
    return deriv;
}

float perdaTripleto(const std::vector<float>& ancora, const std::vector<float>& positiva, const std::vector<float>& negativa, float margem = 1.0f) {
    float distPos = 0.0f;
    float distNeg = 0.0f;
    for(size_t i = 0; i < ancora.size(); i++) {
        distPos += std::pow(ancora[i] - positiva[i], 2);
        distNeg += std::pow(ancora[i] - negativa[i], 2);
    }
    return std::max(0.0f, distPos - distNeg + margem);
}

float contrastivaPerda(const std::vector<float>& saida1, const std::vector<float>& saida2, int rotulo, float margem = 1.0f) {
    float distancia = 0.0f;
    for(size_t i = 0; i < saida1.size(); i++) distancia += std::pow(saida1[i] - saida2[i], 2);
    
    if(rotulo == 1) return distancia;
    else return std::max(0.0f, margem - std::sqrt(distancia));
}

// funções de saida:
std::vector<float> softmax(const std::vector<float>& arr, float temp = 1.0f) {
    // encontra o maior valor para evitar overflow
    float max = *std::max_element(arr.begin(), arr.end());
    
    // calcula exponenciais
    std::vector<float> exps(arr.size());
    float soma = 0.0f;
    for(size_t i = 0; i < arr.size(); ++i) {
        exps[i] = std::exp((arr[i] - max) / temp);
        soma += exps[i];
    }
    // evita divisão por zero
    if(soma < 1e-6f) soma = 1e-6f;
    // normalizar
    for(size_t i = 0; i < exps.size(); ++i) exps[i] /= soma;
    
    return exps;
}

std::vector<float> derivadaSoftmax(const std::vector<float>& arr, const std::vector<float>& gradSaida) {
    float soma = 0.0f;
    
    // calcula soma de gradSaida[i] * arr[i]
    for(size_t j = 0; j < gradSaida.size(); ++j) soma += gradSaida[j] * arr[j];
    
    // calcula derivada
    std::vector<float> res(arr.size());
    for(size_t i = 0; i < arr.size(); ++i) res[i] = arr[i] * (gradSaida[i] - soma);
    
    return res;
}

std::vector<std::vector<float>> softmaxLote(const std::vector<std::vector<float>>& m, float temp = 1.0f) {
    std::vector<std::vector<float>> res(m.size());
    for(size_t i = 0; i < m.size(); ++i) res[i] = softmax(m[i], temp);
    return res;
}

int argmax(const std::vector<float>& v) {
    if(v.empty()) return -1; // caso o vetor esteja vazio
    return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

std::vector<float> addRuido(const std::vector<float>& v, float intenso = 0.01f) {
    // configura gerador de números aleatórios
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-intenso, intenso);
    
    std::vector<float> res(v.size());
    for(size_t i = 0; i < v.size(); ++i) res[i] = v[i] + dis(gen);
    return res;
}

void testeU() {
    std::cout << "\n=== TESTES SAIDA ===\n\n";
    std::vector<float> entrada = {1.0f, 2.0f, 3.0f};
    std::vector<float> saida_softmax = softmax(entrada, 1.0f);
    std::cout << "Softmax: ";
    for(float x : saida_softmax) std::cout << x << " ";
    std::cout << std::endl;
    std::vector<float> grad = {0.1f, 0.2f, 0.3f};
    std::vector<float> saida_derivada = derivadaSoftmax(saida_softmax, grad);
    std::cout << "Derivada Softmax: ";
    for(float x : saida_derivada) std::cout << x << " ";
    std::cout << std::endl;
    std::vector<std::vector<float>> matriz = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    std::vector<std::vector<float>> saida_lote = softmaxLote(matriz, 1.0f);
    std::cout << "Softmax Lote:\n";
    for(const auto& linha : saida_lote) {
        for(float x : linha) std::cout << x << " ";
        std::cout << std::endl;
    }
    std::cout << "Argmax: " << argmax(entrada) << std::endl;
    std::vector<float> saida_ruido = addRuido(entrada, 0.01f);
    std::cout << "Add Ruido: ";
    for(float x : saida_ruido) std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "\n=== TESTES ERRO ===\n\n";
    std::vector<float> saida = {0.8f, 0.2f, 0.5f};
    std::vector<float> esperado = {1.0f, 0.0f, 0.5f};
    std::vector<float> y = {1.0f, 0.0f, 0.0f};
    std::vector<float> yChapeu = {0.7f, 0.2f, 0.1f};
    std::vector<float> ancora = {0.5f, 0.5f};
    std::vector<float> positiva = {0.6f, 0.6f};
    std::vector<float> negativa = {0.4f, 0.4f};
    std::vector<float> saida1 = {0.1f, 0.9f};
    std::vector<float> saida2 = {0.2f, 0.8f};
    std::cout << "erroAbsolutoMedio: " << erroAbsolutoMedio(saida, esperado) << std::endl;
    std::cout << "erroQuadradoEsperado: " << erroQuadradoEsperado(saida, esperado) << std::endl;
    auto derivErro = derivadaErro(saida, esperado);
    std::cout << "derivadaErro: [";
    for(float val : derivErro) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "entropiaCruzada: " << entropiaCruzada(y, yChapeu) << std::endl;
    auto derivEntropia = derivadaEntropiaCruzada(y, yChapeu);
    std::cout << "derivadaEntropiaCruzada: [";
    for(float val : derivEntropia) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "huberPerda: " << huberPerda(saida, esperado) << std::endl;
    auto derivHuber = derivadaHuber(saida, esperado);
    std::cout << "derivadaHuber: [";
    for(float val : derivHuber) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "perdaTripleto: " << perdaTripleto(ancora, positiva, negativa) << std::endl;
    std::cout << "contrastivaPerda (rotulo=1): " << contrastivaPerda(saida1, saida2, 1) << std::endl;
    std::cout << "contrastivaPerda (rotulo=0): " << contrastivaPerda(saida1, saida2, 0) << std::endl;
    std::cout << "\n=== TESTES REGULARIZAÇÃO E MÉTRICAS ===\n\n";
    std::vector<std::vector<float>> pesos = {{1.0f, -2.0f, 0.0f}, {3.0f, -4.0f, 5.0f}};
    std::vector<std::vector<float>> tensor = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<float> vetor = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<std::vector<float>> saidas = {{0.8f, 0.1f, 0.1f}, {0.2f, 0.7f, 0.1f}};
    std::vector<std::vector<float>> esperados = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    std::vector<std::vector<int>> matrizConfusao = {{5, 2}, {1, 8}};
    std::vector<float> p = {0.4f, 0.6f};
    std::vector<float> q = {0.5f, 0.5f};
    std::vector<float> pontos = {0.8f, 0.6f, 0.4f, 0.2f};
    std::vector<int> rotulos = {1, 0, 1, 0};
    auto l1 = regularL1(pesos, 0.1f);
    std::cout << "L1 Regularization: [";
    for(const auto& linha : l1) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "]\n";
    auto l2 = regularL2(pesos, 0.1f);
    std::cout << "L2 Regularization: [";
    for(const auto& linha : l2) {
        for(float val : linha) std::cout << val << " ";
    }
    std::cout << "]\n";
    auto dropoutRes = dropout(tensor, 0.3f);
    std::cout << "Dropout: [";
    for(const auto& linha : dropoutRes) for (float val : linha) std::cout << val << " ";
    std::cout << "]\n";
    auto normEnt = normEntrada(vetor);
    std::cout << "Normalização Entrada: [";
    for (float val : normEnt) std::cout << val << " ";
    std::cout << "]\n";
    auto normZ = normZPonto(vetor);
    std::cout << "Normalização Z: [";
    for(float val : normZ) std::cout << val << " ";
    std::cout << "]\n";
    std::cout << "Acurácia: " << acuracia(saidas, esperados) << "\n";
    std::cout << "Precisão: " << precisao(matrizConfusao) << "\n";
    std::cout << "Recall: " << recall(matrizConfusao) << "\n";
    std::cout << "F1-ponto: " << f1Ponto(matrizConfusao) << "\n";
    std::cout << "MSE: " << mse({1.0f, 2.0f}, {1.5f, 1.8f}) << "\n";
    std::cout << "KL Divergence: " << klDivergencia(p, q) << "\n";
    std::cout << "ROC AUC: " << rocAuc(pontos, rotulos) << "\n";
}