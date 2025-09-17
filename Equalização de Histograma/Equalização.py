import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from google.colab import files
from skimage import data
import warnings
warnings.filterwarnings('ignore')

# Configuração do estilo dos gráficos
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)

def plot_images(original, equalized, title_original, title_equalized):
    """Função para plotar imagens original e equalizada lado a lado"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Imagem original
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title(title_original, fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Histograma original
    axes[0, 1].hist(original.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[0, 1].set_title('Histograma Original', fontsize=12)
    axes[0, 1].set_xlabel('Intensidade de Pixel')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].grid(True, alpha=0.3)
    
    # CDF original
    hist_orig, _ = np.histogram(original.flatten(), 256, [0, 256])
    cdf_orig = hist_orig.cumsum()
    cdf_orig_normalized = cdf_orig * float(hist_orig.max()) / cdf_orig.max()
    axes[0, 2].plot(cdf_orig_normalized, color='red', linewidth=2)
    axes[0, 2].set_title('CDF Original', fontsize=12)
    axes[0, 2].set_xlabel('Intensidade de Pixel')
    axes[0, 2].set_ylabel('CDF')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Imagem equalizada
    axes[1, 0].imshow(equalized, cmap='gray')
    axes[1, 0].set_title(title_equalized, fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Histograma equalizado
    axes[1, 1].hist(equalized.flatten(), bins=256, range=[0, 256], color='green', alpha=0.7)
    axes[1, 1].set_title('Histograma Equalizado', fontsize=12)
    axes[1, 1].set_xlabel('Intensidade de Pixel')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].grid(True, alpha=0.3)
    
    # CDF equalizado
    hist_eq, _ = np.histogram(equalized.flatten(), 256, [0, 256])
    cdf_eq = hist_eq.cumsum()
    cdf_eq_normalized = cdf_eq * float(hist_eq.max()) / cdf_eq.max()
    axes[1, 2].plot(cdf_eq_normalized, color='purple', linewidth=2)
    axes[1, 2].set_title('CDF Equalizado', fontsize=12)
    axes[1, 2].set_xlabel('Intensidade de Pixel')
    axes[1, 2].set_ylabel('CDF')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def equalize_histogram(image):
    """Função para equalizar o histograma de uma imagem"""
    # Calcula o histograma
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    
    # Calcula a CDF (Função de Distribuição Cumulativa)
    cdf = hist.cumsum()
    
    # Normaliza a CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype('uint8')
    
    # Aplica a equalização
    equalized = cdf_normalized[image]
    
    return equalized

print("=" * 80)
print("EQUALIZAÇÃO DE HISTOGRAMA - PROCESSAMENTO DE IMAGENS")
print("=" * 80)

# Primeira imagem: Imagem de exemplo do skimage (moeda)
print("\n PRIMEIRA IMAGEM: MOEDA")
print("-" * 50)

# Carrega a primeira imagem
image1 = data.coins()
image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1

# Equaliza o histograma
equalized1 = equalize_histogram(image1_gray)

# Exibe informações da imagem
print(f"Dimensões da imagem: {image1_gray.shape}")
print(f"Tipo de dados: {image1_gray.dtype}")
print(f"Valor mínimo: {image1_gray.min()}, Valor máximo: {image1_gray.max()}")

# Plota os resultados
plot_images(image1_gray, equalized1, 'Imagem Original - Moeda', 'Imagem Equalizada - Moeda')

# Segunda imagem: Imagem de exemplo do skimage (câmera)
print("\n SEGUNDA IMAGEM: CÂMERA")
print("-" * 50)

# Carrega a segunda imagem
image2 = data.camera()
image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2

# Equaliza o histograma
equalized2 = equalize_histogram(image2_gray)

# Exibe informações da imagem
print(f"Dimensões da imagem: {image2_gray.shape}")
print(f"Tipo de dados: {image2_gray.dtype}")
print(f"Valor mínimo: {image2_gray.min()}, Valor máximo: {image2_gray.max()}")

# Plota os resultados
plot_images(image2_gray, equalized2, 'Imagem Original - Câmera', 'Imagem Equalizada - Câmera')

# Comparação estatística
print("\n COMPARAÇÃO ESTATÍSTICA")
print("-" * 50)

def print_stats(image, title, equalized=False):
    """Função para imprimir estatísticas da imagem"""
    color = "🟢" if equalized else "🔵"
    print(f"{color} {title}:")
    print(f"   Média: {image.mean():.2f}")
    print(f"   Desvio Padrão: {image.std():.2f}")
    print(f"   Variância: {image.var():.2f}")
    print(f"   Entropia: {-np.sum((np.histogram(image, bins=256)[0]/image.size) * \
          np.log2(np.histogram(image, bins=256)[0]/image.size + 1e-10)):.2f}")

print_stats(image1_gray, "Moeda Original")
print_stats(equalized1, "Moeda Equalizada", True)
print()
print_stats(image2_gray, "Câmera Original")
print_stats(equalized2, "Câmera Equalizada", True)

print("\n" + "=" * 80)
print("PROCESSO CONCLUÍDO!")
print("=" * 80)
