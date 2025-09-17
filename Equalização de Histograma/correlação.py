import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform
from scipy import stats
import seaborn as sns
from google.colab.patches import cv2_imshow

# Configuração do estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
np.set_printoptions(precision=4, suppress=True)
sns.set_palette("viridis")

def carregar_imagens():
    """Carrega e prepara duas imagens para análise de correlação"""

    # Primeira imagem: Edifício (mais estruturada)
    edificio = data.brick()
    # Verifica se a imagem tem 3 canais (colorida) antes de converter para tons de cinza
    if len(edificio.shape) == 3:
        edificio = cv2.cvtColor(edificio, cv2.COLOR_RGB2GRAY)
    edificio = transform.resize(edificio, (256, 256), anti_aliasing=True)
    edificio = (edificio * 255).astype(np.uint8)

    # Segunda imagem: Textura de folhas (mais aleatória)
    folhas = data.grass()
    # Verifica se a imagem tem 3 canais (colorida) antes de converter para tons de cinza
    if len(folhas.shape) == 3:
        folhas = cv2.cvtColor(folhas, cv2.COLOR_RGB2GRAY)
    folhas = transform.resize(folhas, (256, 256), anti_aliasing=True)
    folhas = (folhas * 255).astype(np.uint8)

    return edificio, folhas

def calcular_correlacoes(imagem1, imagem2, nome1, nome2):
    """Calcula várias medidas de correlação entre duas imagens"""

    # Garantir que as imagens tenham o mesmo tamanho
    if imagem1.shape != imagem2.shape:
        min_shape = (min(imagem1.shape[0], imagem2.shape[0]),
                     min(imagem1.shape[1], imagem2.shape[1]))
        imagem1 = cv2.resize(imagem1, min_shape[::-1])
        imagem2 = cv2.resize(imagem2, min_shape[::-1])

    # Achatar as imagens para análise
    flat1 = imagem1.flatten()
    flat2 = imagem2.flatten()

    # Diferentes medidas de correlação
    correlacao_pearson = np.corrcoef(flat1, flat2)[0, 1]
    correlacao_spearman = stats.spearmanr(flat1, flat2).statistic
    correlacao_kendall = stats.kendalltau(flat1, flat2).statistic

    # Covariância
    covariancia = np.cov(flat1, flat2)[0, 1]

    # MSE (Mean Squared Error)
    mse = np.mean((flat1 - flat2) ** 2)

    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # Coeficiente de determinação R²
    r_squared = correlacao_pearson ** 2

    # Informação mútua
    hist_2d, _, _ = np.histogram2d(flat1, flat2, bins=50)
    # Normaliza o histograma para obter probabilidades
    hist_2d_norm = hist_2d / np.sum(hist_2d)
    # Calcula a entropia conjunta
    entropia_conjunta = -np.sum(hist_2d_norm * np.log2(hist_2d_norm + 1e-10))

    # Calcula as entropias marginais
    hist1, _ = np.histogram(flat1, bins=50)
    hist1_norm = hist1 / np.sum(hist1)
    entropia1 = -np.sum(hist1_norm * np.log2(hist1_norm + 1e-10))

    hist2, _ = np.histogram(flat2, bins=50)
    hist2_norm = hist2 / np.sum(hist2)
    entropia2 = -np.sum(hist2_norm * np.log2(hist2_norm + 1e-10))

    # Informação Mútua = Entropia(X) + Entropia(Y) - Entropia(X, Y)
    info_mutua = entropia1 + entropia2 - entropia_conjunta


    return {
        'Pearson': correlacao_pearson,
        'Spearman': correlacao_spearman,
        'Kendall': correlacao_kendall,
        'Covariância': covariancia,
        'MSE': mse,
        'PSNR': psnr,
        'R²': r_squared,
        'Info Mútua': info_mutua,
        'imagem1': imagem1,
        'imagem2': imagem2,
        'nome1': nome1,
        'nome2': nome2
    }

def plotar_analise_correlacao(resultados):
    """Plota análise completa da correlação entre imagens"""

    fig = plt.figure(figsize=(20, 15))

    # Layout dos subplots
    gs = fig.add_gridspec(3, 4)

    # 1. Imagens originais
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(resultados['imagem1'], cmap='gray')
    ax1.set_title(f'{resultados["nome1"]}\n({resultados["imagem1"].shape})',
                 fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(resultados['imagem2'], cmap='gray')
    ax2.set_title(f'{resultados["nome2"]}\n({resultados["imagem2"].shape})',
                 fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 2. Histogramas
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.hist(resultados['imagem1'].flatten(), bins=50, alpha=0.7,
             label=resultados['nome1'], color='blue')
    ax3.hist(resultados['imagem2'].flatten(), bins=50, alpha=0.7,
             label=resultados['nome2'], color='red')
    ax3.set_title('Histogramas Comparativos', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Intensidade de Pixel')
    ax3.set_ylabel('Frequência')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3. Scatter plot da correlação
    ax4 = fig.add_subplot(gs[1, :2])
    sample_size = min(10000, len(resultados['imagem1'].flatten())) # Aumenta o sample size
    indices = np.random.choice(len(resultados['imagem1'].flatten()), sample_size, replace=False)

    ax4.scatter(resultados['imagem1'].flatten()[indices],
               resultados['imagem2'].flatten()[indices],
               alpha=0.6, s=10, color='green')
    ax4.set_xlabel(f'Intensidade - {resultados["nome1"]}')
    ax4.set_ylabel(f'Intensidade - {resultados["nome2"]}')
    ax4.set_title(f'Diagrama de Dispersão\nCorrelação Pearson: {resultados["Pearson"]:.4f}',
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 4. Heatmap 2D da distribuição conjunta
    ax5 = fig.add_subplot(gs[1, 2:])
    hist, xedges, yedges = np.histogram2d(resultados['imagem1'].flatten(),
                                         resultados['imagem2'].flatten(),
                                         bins=50) # Aumenta o número de bins
    im = ax5.imshow(hist.T, origin='lower', cmap='hot',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto') # Adiciona aspect='auto'
    ax5.set_xlabel(f'Intensidade - {resultados["nome1"]}')
    ax5.set_ylabel(f'Intensidade - {resultados["nome2"]}')
    ax5.set_title('Distribuição Conjunta (Heatmap 2D)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Frequência')

    # 5. Bar plot das medidas de correlação
    ax6 = fig.add_subplot(gs[2, :])
    medidas = ['Pearson', 'Spearman', 'Kendall', 'R²']
    valores = [resultados['Pearson'], resultados['Spearman'],
              resultados['Kendall'], resultados['R²']]

    bars = ax6.bar(medidas, valores, color=['blue', 'orange', 'green', 'red'])
    ax6.set_ylabel('Valor da Correlação')
    ax6.set_title('Comparação das Medidas de Correlação', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{valor:.4f}', ha='center', va='bottom', fontweight='bold')

    # Ajustar layout
    plt.tight_layout()
    plt.show()

    return fig

def exibir_metricas_detalhadas(resultados):
    """Exibe as métricas de correlação em formato de tabela"""

    print(" MÉTRICAS DE CORRELAÇÃO DETALHADAS")
    print("=" * 60)
    print(f"{'Métrica':<15} {'Valor':<15} {'Interpretação':<30}")
    print("=" * 60)

    # Pearson
    pearson = resultados['Pearson']
    if abs(pearson) > 0.7:
        interp = "Forte correlação"
    elif abs(pearson) > 0.3:
        interp = "Correlação moderada"
    else:
        interp = "Fraca correlação"
    print(f"{'Pearson':<15} {pearson:<15.4f} {interp:<30}")

    # Spearman
    spearman = resultados['Spearman']
    print(f"{'Spearman':<15} {spearman:<15.4f} {'Correlação de postos':<30}")

    # Kendall
    kendall = resultados['Kendall']
    print(f"{'Kendall':<15} {kendall:<15.4f} {'Concordância de pares':<30}")

    # R²
    r2 = resultados['R²']
    print(f"{'R²':<15} {r2:<15.4f} {f'{r2*100:.1f}% variância explicada':<30}")

    # MSE e PSNR
    print(f"{'MSE':<15} {resultados['MSE']:<15.2f} {'Erro quadrático médio':<30}")
    print(f"{'PSNR':<15} {resultados['PSNR']:<15.2f} {'dB (qualidade)':<30}")

    # Covariância
    print(f"{'Covariância':<15} {resultados['Covariância']:<15.2f} {'Variação conjunta':<30}")

    # Informação Mútua
    print(f"{'Info Mútua':<15} {resultados['Info Mútua']:<15.2f} {'Informação compartilhada':<30}")

    print("=" * 60)

# Execução principal
print(" ANÁLISE DE CORRELAÇÃO ENTRE IMAGENS")
print("=" * 80)

# Carregar imagens
print("CARREGANDO IMAGENS...")
edificio, folhas = carregar_imagens()

print(f" Imagens carregadas:")
print(f"   • Edifício: {edificio.shape}")
print(f"   • Folhas: {folhas.shape}")

# Calcular correlações
print("\n CALCULANDO CORRELAÇÕES...")
resultados = calcular_correlacoes(edificio, folhas, "Edifício", "Textura de Folhas")

# Exibir métricas
exibir_metricas_detalhadas(resultados)

# Plotar análise completa
print("\n GERANDO VISUALIZAÇÕES...")
figura = plotar_analise_correlacao(resultados)

# Análise interpretativa
print("\n INTERPRETAÇÃO DOS RESULTADOS:")
print("=" * 80)

pearson = resultados['Pearson']
if abs(pearson) > 0.7:
    print("CORRELAÇÃO FORTE: As imagens têm padrões similares")
elif abs(pearson) > 0.3:
    print("CORRELAÇÃO MODERADA: Alguma similaridade nos padrões")
else:
    print("CORRELAÇÃO FRACA: Imagens com padrões distintos")

print(f"\n RESUMO:")
print(f"   * Correlação Pearson: {pearson:.4f}")
print(f"   * Variância explicada (R²): {resultados['R²']*100:.1f}%")
print(f"   * MSE: {resultados['MSE']:.2f}")
print(f"   * PSNR: {resultados['PSNR']:.2f} dB")
print(f"   * Informação Mútua: {resultados['Info Mútua']:.2f} bits")


print("\n" + "=" * 80)
print("ANÁLISE DE CORRELAÇÃO CONCLUÍDA! ")
print("=" * 80)

# Exibir as imagens individualmente também
print("\n  IMAGENS INDIVIDUAIS:")
print("\n1. IMAGEM DO EDIFÍCIO:")
plt.figure(figsize=(8, 6))
plt.imshow(edificio, cmap='gray')
plt.title('Edifício - Imagem Estruturada', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

print("\n2. IMAGEM DA TEXTURA DE FOLHAS:")
plt.figure(figsize=(8, 6))
plt.imshow(folhas, cmap='gray')
plt.title('Textura de Folhas - Imagem Aleatória', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()
