# 🤖 Nyxie: Protogen Telegram Chatbot 🌟

## 🌈 Giriş

**Nyxie**, Stixyie tarafından geliştirilen gelişmiş bir Protogen AI Telegram chatbot'udur. Yapay zeka teknolojisini kullanarak kullanıcılarla etkileşime giren, zamansal ve bağlamsal olarak duyarlı bir asistantır türkçeye odaklı yapılmıştır ama diğer dillerde belli bir seviyede destek sağlar ama asıl amaçı türkçe konusmayı desteklemektir.

### 🤔 Nyxie Nedir?

Nyxie, sadece bir chatbot değil, aynı zamanda:
- 🧠 Gelişmiş yapay zeka teknolojisi ile çalışan bir dijital arkadaş
- 🌍 Çoklu dil desteği olan bir iletişim asistanı
- 🕰️ Zamansal ve mekânsal farkındalığa sahip bir AI
- 🌈 Dinamik kişilik profili ile etkileşime giren bir asistan
- 🤖 Protogen kimliğine sahip, duygusal ve yaratıcı bir AI

**Nyxie'nin Benzersiz Özellikleri:**
- Her etkileşimi benzersiz ve kişiselleştirilmiş kılan dinamik zaman bağlamı
- Günün saatine, mevsimine ve kullanıcının yerel bağlamına göre değişen kişilik
- Kullanıcının dilini ve tercihlerini anlayan ve uyarlayan akıllı bir sistem

## 🚀 Özellikler

### 1. 💬 Gelişmiş Konuşma Yeteneği
- Dinamik ve bağlamsal yanıtlar
- Kullanıcı tercihlerini öğrenme ve hatırlama
- Çoklu dil desteği (Türkçe, İngilizce ve diğer diller)
- Doğal dil işleme ile dil ve ayar tespiti
- Otomatik emoji ekleme ve yanıt zenginleştirme
- Kullanıcı dilini ve tercihlerini otomatik algılama

### 2. 🕒 Zamansal Kişilik Uyarlaması
- Günün saatine göre dinamik kişilik profili
- Mevsim, hafta günü ve günün periyoduna göre yanıt uyarlama
- Kullanıcının yerel saat dilimini ve zamanını algılama
- Günün saatine, mevsimine ve özel günlere göre kişilik değişimi
- Hafta içi/hafta sonu ve tatil günlerinde farklı davranış modları

### 3. 🖼️ Görüntü ve Video İşleme
- Gönderilen görüntüleri ve videoları analiz etme
- Google Cloud Vision API ile görüntü tanıma
- Multimedya içeriği hakkında açıklama üretme

### 4. 🧠 Kullanıcı Hafızası
- Kullanıcı tercihlerini ve geçmiş etkileşimlerini kaydetme
- Maksimum 2 milyon token'a kadar konuşma geçmişi
- Her kullanıcı için ayrı JSON hafıza dosyaları
- Otomatik dil ve zaman dilimi tespiti
- Güvenli ve şifrelenmiş kullanıcı verileri
- Dinamik tercih ve ayar yönetimi
- Konuşma bağlamını ve kullanıcı tercihlerini koruma

### 5. 🌐 Akıllı Web Arama
- Gemini AI ile dinamik web arama
- Kullanıcı sorgularını akıllıca yorumlama
- Web arama sonuçlarını analiz etme ve özetleme
- Çoklu kaynaklardan bilgi toplama
- Arama sonuçlarını kullanıcı diline çevirme
- Güvenilir ve güncel bilgi sağlama

## 🛠️ Gereksinimler

### Yazılım Gereksinimleri
- Python 3.8+
- pip paket yöneticisi

### Gerekli Kütüphaneler
- python-telegram-bot
- google-generativeai
- python-dotenv
- requests
- geopy
- timezonefinder
- emoji
- langdetect
- Pillow
- httpx
- google-cloud-vision

## 🔧 Kurulum

### 1. Depoyu Klonlama
```bash
git clone https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v4-main.git
cd Nyxie-Protogen-Chatbot-Telegram-v4-main
```

### 2. Sanal Ortam Oluşturma
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

## 🔐 Konfigürasyon

### Gerekli API Anahtarları
`.env` dosyasında aşağıdaki API anahtarlarını yapılandırın:
- `TELEGRAM_TOKEN`: Telegram Bot Token you need to get this token from here: https://t.me/BotFather
- `GEMINI_API_KEY`: Google Ai Studio API Key you need to get this key from here: https://aistudio.google.com/apikey

### Örnek `.env` Dosyası
```
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
```

## 🚀 Kullanım

### Bot'u Başlatma
```bash
python bot.py
```

### Telegram'da Kullanım
1. Bot'a `/start` komutu ile başlayın
2. Mesaj, görüntü veya video gönderin
3. Sohbet için bot ile etkileşime geçin

## 🛡️ Güvenlik

- Kullanıcı verileri şifrelenmiş JSON dosyalarında saklanır
- Maksimum token sınırlaması ile bellek yönetimi
- Hassas bilgilerin loglanmaması

## 🤝 Destek

### Sorun Bildirim
- GitHub Issues: [Proje Sayfası](https://github.com/stixyie/Nyxie-Protogen-Chatbot-Telegram-v2-main/issues)

### Katkıda Bulunma
1. Projeyi forklayın
2. Yeni bir branch oluşturun
3. Değişikliklerinizi yapın
4. Pull Request açın

## 📄 Lisans

Bu proje GPL-3.0 Lisansı altında yayınlanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🌟 Teşekkür

- **Stixyie**: Proje yaratıcısı ve baş geliştirici
- **Google**: Gemini ve Cloud Vision API'ları

---

**Not**: Nyxie, sürekli gelişen bir AI projesidir. Geri bildirimleriniz ve katkılarınız çok değerlidir! 🚀
