import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy import stats
import pandas as pd

# Configuração do estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
np.set_printoptions(precision=4, suppress=True)

def calcular_estatisticas(imagem, nome_imagem):
    """Calcula todas as medidas estatísticas para uma imagem"""
    
    # Estatísticas básicas
    media = np.mean(imagem)
    mediana = np.median(imagem)
    moda = stats.mode(imagem.flatten(), keepdims=True).mode[0]
    variancia = np.var(imagem)
    desvio_padrao = np.std(imagem)
    amplitude = np.ptp(imagem)  # Peak to Peak
    minimo = np.min(imagem)
    maximo = np.max(imagem)
    
    # Medidas de forma
    skewness = stats.skew(imagem.flatten())
    kurtosis = stats.kurtosis(imagem.flatten())
    
    # Quartis e percentis
    q1 = np.percentile(imagem, 25)
    q2 = np.percentile(imagem, 50)  # Mediana
    q3 = np.percentile(imagem, 75)
    iqr = q3 - q1
    
    # Entropia
    hist, _ = np.histogram(imagem, bins=256, range=(0, 256))
    prob = hist / hist.sum()
    entropia = -np.sum(prob * np.log2(prob + 1e-10))  # +1e-10 para evitar log(0)
    
    # Momentos estatísticos
    momento_ordem1 = np.mean(imagem)
    momento_ordem2 = np.mean(imagem**2)
    momento_ordem3 = np.mean(imagem**3)
    momento_ordem4 = np.mean(imagem**4)
    
    # Momentos centrais
    momento_central_2 = np.mean((imagem - media)**2)  # Variância
    momento_central_3 = np.mean((imagem - media)**3)
    momento_central_4 = np.mean((imagem - media)**4)
    
    return {
        'Imagem': nome_imagem,
        'Média': media,
        'Mediana': mediana,
        'Moda': moda,
        'Variância': variancia,
        'Desvio Padrão': desvio_padrao,
        'Amplitude': amplitude,
        'Mínimo': minimo,
        'Máximo': maximo,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Q1 (25%)': q1,
        'Q2 (Mediana)': q2,
        'Q3 (75%)': q3,
        'IQR': iqr,
        'Entropia': entropia,
        'Momento Ordem 1': momento_ordem1,
        'Momento Ordem 2': momento_ordem2,
        'Momento Ordem 3': momento_ordem3,
        'Momento Ordem 4': momento_ordem4,
        'Momento Central 2': momento_central_2,
        'Momento Central 3': momento_central_3,
        'Momento Central 4': momento_central_4
    }

def plotar_comparacao_histogramas(imagem1, nome1, imagem2, nome2):
    """Plota histogramas comparativos das duas imagens"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histograma da primeira imagem
    ax1.hist(imagem1.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'Histograma - {nome1}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intensidade de Pixel')
    ax1.set_ylabel('Frequência')
    ax1.grid(True, alpha=0.3)
    
    # Histograma da segunda imagem
    ax2.hist(imagem2.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title(f'Histograma - {nome2}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intensidade de Pixel')
    ax2.set_ylabel('Frequência')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def exibir_tabela_estatisticas(estatisticas1, estatisticas2):
    """Exibe as estatísticas em formato de tabela comparativa"""
    
    # Cria DataFrame comparativo
    df_comparativo = pd.DataFrame({
        'Estatística': list(estatisticas1.keys())[1:],  # Exclui o nome da imagem
        estatisticas1['Imagem']: list(estatisticas1.values())[1:],
        estatisticas2['Imagem']: list(estatisticas2.values())[1:]
    })
    
    print(" TABELA COMPARATIVA DE ESTATÍSTICAS")
    print("=" * 80)
    print(df_comparativo.to_string(index=False))
    print("=" * 80)

# Carregar as imagens
print("CARREGANDO IMAGENS PARA ANÁLISE ESTATÍSTICA")
print("=" * 80)

# Primeira imagem: Moeda
imagem_moeda = data.coins()
if len(imagem_moeda.shape) == 3:
    imagem_moeda = cv2.cvtColor(imagem_moeda, cv2.COLOR_RGB2GRAY)

# Segunda imagem: Câmera
imagem_camera = data.camera()
if len(imagem_camera.shape) == 3:
    imagem_camera = cv2.cvtColor(imagem_camera, cv2.COLOR_RGB2GRAY)

print(f"Dimensões das imagens:")
print(f"Moeda: {imagem_moeda.shape}")
print(f"Câmera: {imagem_camera.shape}")
print()

# Calcular estatísticas
print(" CALCULANDO MEDIDAS ESTATÍSTICAS...")
print()

estatisticas_moeda = calcular_estatisticas(imagem_moeda, "Moeda")
estatisticas_camera = calcular_estatisticas(imagem_camera, "Câmera")

# Exibir tabela comparativa
exibir_tabela_estatisticas(estatisticas_moeda, estatisticas_camera)

# Plotar histogramas comparativos
print("\n HISTOGRAMAS COMPARATIVOS")
plotar_comparacao_histogramas(imagem_moeda, "Moeda", imagem_camera, "Câmera")

# Análise interpretativa
print("\n ANÁLISE INTERPRETATIVA DAS ESTATÍSTICAS")
print("=" * 80)

print(" 1 TENDÊNCIA CENTRAL:")
print(f"   * Moeda: Média = {estatisticas_moeda['Média']:.2f}, Mediana = {estatisticas_moeda['Mediana']:.2f}")
print(f"   * Câmera: Média = {estatisticas_camera['Média']:.2f}, Mediana = {estatisticas_camera['Mediana']:.2f}")
print("   → Ambas têm médias e medianas próximas, indicando distribuição balanceada")

print("\n 2 DISPERSÃO:")
print(f"   * Moeda: DP = {estatisticas_moeda['Desvio Padrão']:.2f}, Variância = {estatisticas_moeda['Variância']:.2f}")
print(f"   * Câmera: DP = {estatisticas_camera['Desvio Padrão']:.2f}, Variância = {estatisticas_camera['Variância']:.2f}")
print("   → A câmera tem maior variabilidade de intensidades")

print("\n 3  FORMA DA DISTRIBUIÇÃO:")
print(f"  * Moeda: Skewness = {estatisticas_moeda['Skewness']:.3f} (ligeiramente assimétrica)")
print(f"  * Câmera: Skewness = {estatisticas_camera['Skewness']:.3f} (próximo de simétrica)")
print(f"  * Moeda: Kurtosis = {estatisticas_moeda['Kurtosis']:.3f}")
print(f"  * Câmera: Kurtosis = {estatisticas_camera['Kurtosis']:.3f}")
print("   → Kurtosis negativo indica distribuição mais achatada que a normal")

print("\n 4  ENTROPIA (INFORMAÇÃO):")
print(f" * Moeda: Entropia = {estatisticas_moeda['Entropia']:.3f} bits")
print(f" * Câmera: Entropia = {estatisticas_camera['Entropia']:.3f} bits")
print("   → Maior entropia indica mais informação/complexidade na imagem")

print("\n 5  INTERVALO INTERQUARTIL (IQR):")
print(f" * Moeda: IQR = {estatisticas_moeda['IQR']:.2f}")
print(f" * Câmera: IQR = {estatisticas_camera['IQR']:.2f}")
print("   → IQR mostra a dispersão dos 50% centrais dos dados")

# Exibir as imagens originais
print("\n IMAGENS ORIGINAIS")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(imagem_moeda, cmap='gray')
ax1.set_title('Imagem da Moeda', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2.imshow(imagem_camera, cmap='gray')
ax2.set_title('Imagem da Câmera', fontsize=14, fontweight='bold')
ax2.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ANÁLISE ESTATÍSTICA CONCLUÍDA! ")
print("=" * 80)
