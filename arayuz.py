#  -*- coding: utf-8 -*-
from tkinter import *
import tweepy
import preprocessor as p
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from sklearn.cross_validation import KFold
nltk.download('punkt')
nltk.download('wordnet')

x_egitim = []
y_egitim = []
x_test = []
y_test = []
egitim_basari_toplam = 0
test_basari_toplam = 0

dil = ''
sayac_p = 0
sayac_n = 0
label_konum = 260
satir_limit = 10000000



def tweetCek():


    file = open('hamveri.txt', 'w', encoding='utf-8')
    consumer_key = '6gXdcmOfkRiqCll02NxMknMNs'
    consumer_secret = 'dvWXKDlGUnh83bSVUINNBQSYsI8UZbAi3S864FLKGMciJuKoyg'
    access_token = '755976439-aUsWnbuJuw3RXXIgApNHWyZVemosODFCbl6BmdYl'
    access_token_secret = 'b6ioDzWqSAnz1F39M8siH7u5J9hbFbtkpMB2KCxwpDfSh'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    aranacakkonu = konu.get()
    dil_secim = veri.get()
    while (True):
        if (dil_secim == '1'):
            dil = 'tr'
            break
        elif (dil_secim == '2'):
            dil = 'en'
            break
        else:
            isim3 = Label(pencere, text="Geçerli bir seçim yapınız")
            isim3.pack()
    for tweet in tweepy.Cursor(api.search, q=aranacakkonu, count=10, lang=dil).items(20):
        file.write(p.clean(tweet.text))
        file.write('\n')

    def kelime_bol(pos, neg):
        kelime = []
        for file in [pos, neg]:
            with open(file, 'r', encoding='utf-8') as f:
                icerik = f.readlines()
                for l in icerik[:satir_limit]:
                    tum_kelime = word_tokenize(l.lower())
                    kelime += list(tum_kelime)
        kelime = [kelime_bolucu.lemmatize(i) for i in kelime]
        kelime_sayi = Counter(kelime)
        kelime_havuz = []
        for kelimeler in kelime_sayi:
            if 1000 > kelime_sayi[kelimeler] > 50:
                kelime_havuz.append(kelimeler)
        return kelime_havuz

    def etiketle(veri_seti, kelime, sinif_etiket):
        etiketli_vektor = []
        with open(veri_seti, 'r', encoding='utf-8') as f:
            icerik = f.readlines()
            for l in icerik[:satir_limit]:
                mevcut_kelime = word_tokenize(l.lower())
                mevcut_kelime = [kelime_bolucu.lemmatize(i) for i in mevcut_kelime]
                vektor = np.zeros(len(kelime))
                for kelimeler in mevcut_kelime:
                    if kelimeler.lower() in kelime:
                        index_value = kelime.index(kelimeler.lower())
                        vektor[index_value] += 1
                vektor = list(vektor)
                etiketli_vektor.append([vektor, sinif_etiket])
        return etiketli_vektor

    def egitim_test(pos, neg):
        kelime = kelime_bol(pos, neg)
        with open('kelime.pickle', 'wb') as f:
            pickle.dump(kelime, f)
        vektor = []
        vektor += etiketle(pos, kelime, [1, 0])
        vektor += etiketle(neg, kelime, [0, 1])
        random.shuffle(vektor)
        vektor = np.array(vektor)
        index = 0
        egitim_x = []
        egitim_y = []
        test_x = []
        test_y = []
        k = 10
        kf = KFold(len(vektor), n_folds=k)
        for egitim_index, test_index in kf:
            egitim_vektor = []
            test_vektor = []
            for i in range(0, len(egitim_index)):
                temp = egitim_index[i]
                egitim_vektor.append(vektor[temp])
            for i in range(0, len(test_index)):
                temp = test_index[i]
                test_vektor.append(vektor[temp])
            for i in range(0, len(egitim_vektor)):
                egitim_x.append([index, egitim_vektor[i][0]])
                egitim_y.append([index, egitim_vektor[i][1]])
            for i in range(0, len(test_vektor)):
                test_x.append([index, test_vektor[i][0]])
                test_y.append([index, test_vektor[i][1]])
            index = index + 1
        return egitim_x, egitim_y, test_x, test_y, k

    kelime_bolucu = WordNetLemmatizer()

    if (dil == 'tr'):
        egitim_x, egitim_y, test_x, test_y, k = egitim_test('postr.txt', 'negtr.txt')

    if (dil == 'en'):
        egitim_x, egitim_y, test_x, test_y, k = egitim_test('pos.txt', 'neg.txt')

    gizli_katman_dugum = 10
    sinif_sayi = 2
    x = tf.placeholder('float', [None, len(egitim_x[0][1])])
    y = tf.placeholder('float')
    gizli_katman_1 = {
        'agirlik': tf.Variable(tf.truncated_normal([len(egitim_x[0][1]), gizli_katman_dugum], stddev=0.1)),
        'esik': tf.Variable(tf.constant(0.1, shape=[gizli_katman_dugum]))}
    gizli_katman_2 = {
        'agirlik': tf.Variable(tf.truncated_normal([gizli_katman_dugum, gizli_katman_dugum], stddev=0.1)),
        'esik': tf.Variable(tf.constant(0.1, shape=[gizli_katman_dugum]))}
    cikis_katman = {'agirlik': tf.Variable(tf.truncated_normal([gizli_katman_dugum, sinif_sayi], stddev=0.1)),
                    'esik': tf.Variable(tf.constant(0.1, shape=[sinif_sayi]))}

    def sinir_agi(veri):
        katman_1 = tf.add(tf.matmul(veri, gizli_katman_1['agirlik']), gizli_katman_1['esik'])
        katman_1 = tf.nn.relu(katman_1)
        katman_2 = tf.add(tf.matmul(katman_1, gizli_katman_2['agirlik']), gizli_katman_2['esik'])
        katman_2 = tf.nn.relu(katman_2)
        katman_cikti = tf.matmul(katman_2, cikis_katman['agirlik']) + cikis_katman['esik']
        katman_cikti = tf.nn.relu(katman_cikti)
        return katman_cikti

    def agi_egit(x, x_egitim, y_egitim, x_test, y_test, iter):
        global egitim_basari_toplam
        global test_basari_toplam
        tahmin = sinir_agi(x)
        hata = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tahmin, labels=y))
        optimize_et = tf.train.AdamOptimizer(0.001).minimize(hata)
        epoch_sayi = 15
        grup_boyut = 10
        with tf.Session() as oturum:
            oturum.run(tf.global_variables_initializer())
            for epoch in range(epoch_sayi):
                epoch_hata = 0
                i = 0
                while i < len(x_egitim):
                    baslangic = i
                    bitis = i + grup_boyut
                    batch_x = np.array(x_egitim[baslangic:bitis])
                    batch_y = np.array(y_egitim[baslangic:bitis])
                    _, adim_hata = oturum.run([optimize_et, hata], feed_dict={x: batch_x, y: batch_y})
                    epoch_hata += adim_hata
                    i += grup_boyut
            dogruluk = tf.equal(tf.argmax(tahmin, 1), tf.argmax(y, 1))
            yuzde = tf.reduce_mean(tf.cast(dogruluk, 'float'))
            egitim_basari_toplam += yuzde.eval({x: x_egitim, y: y_egitim})
            kaydedici = tf.train.Saver()
            kaydedici.save(oturum, save_path='./kayit/model')
            test_basari_toplam += yuzde.eval({x: x_test, y: y_test})
            if (iter == k - 1):
                egitim_basari_ort = egitim_basari_toplam / k
                test_basari_ort = test_basari_toplam / k
                basari = Label(pencere, text="Başarı oranı(Eğitim):%" + str(round(100 * egitim_basari_ort, 2)),
                               font="Times 15 bold")
                basari.place(x=525, y=200)
                basari_test = Label(pencere, text="Başarı oranı(Test):%" + str(round(100 * test_basari_ort, 2)),
                                    font="Times 15 bold")
                basari_test.place(x=525, y=230)

    def analiz_yap(tweet):
        global sayac_p
        global sayac_n
        tahmin = sinir_agi(x)
        kayit_noktasi = tf.train.latest_checkpoint('./kayit/')
        with open('kelime.pickle', 'rb') as f:
            kelime = pickle.load(f)

        with tf.Session() as oturum:
            oturum.run(tf.initialize_all_variables())
            kaydedici = tf.train.import_meta_graph("{}.meta".format(kayit_noktasi))
            kaydedici.restore(oturum, kayit_noktasi)
            mevcut_kelime = word_tokenize(tweet.lower())
            mevcut_kelime = [kelime_bolucu.lemmatize(i) for i in mevcut_kelime]
            vektor = np.zeros(len(kelime))

            for kelimeler in mevcut_kelime:
                if kelimeler.lower() in kelime:
                    index_value = kelime.index(kelimeler.lower())
                    vektor[index_value] += 1

            vektor = np.array(list(vektor))
            sonuc = (oturum.run(tf.argmax(tahmin.eval(feed_dict={x: [vektor]}), 1)))
            global label_konum
            if sonuc[0] == 0:
                pozitif = Label(pencere, text="Pozitif", font="Times 12 bold", fg='green')
                positif_label = Label(pencere, text=tweet, font="Times 12 bold")
                sayac_p = sayac_p + 1
                pozitif.place(x=230, y=label_konum)
                positif_label.place(x=300, y=label_konum)
                label_konum = label_konum + 20
            elif sonuc[0] == 1:
                negatif = Label(pencere, text="Negatif", font="Times 12 bold", fg='red')
                negatif_label = Label(pencere, text=tweet, font="Times 12 bold")
                sayac_n = sayac_n + 1
                negatif.place(x=230, y=label_konum)
                negatif_label.place(x=300, y=label_konum)
                label_konum = label_konum + 20

    for j in range(0, k):
        for i in range(0, len(egitim_x)):
            if (egitim_x[i][0] != j and egitim_y[i][0] != j):
                continue
            x_egitim.append(egitim_x[i][1])
            y_egitim.append(egitim_y[i][1])
        for i in range(0, len(test_x)):
            if (test_y[i][0] != j and test_x[i][0] != j):
                continue
            x_test.append(test_x[i][1])
            y_test.append(test_y[i][1])
        agi_egit(x, x_egitim, y_egitim, x_test, y_test, j)
        x_egitim.clear()
        y_egitim.clear()
        x_test.clear()
        y_test.clear()
    file = open('hamveri.txt', 'r', encoding='utf-8')
    tweettemp = file.read()
    tweettemp = tweettemp.split('\n')
    tweet = []
    for i in range(0, len(tweettemp)):
        if (tweettemp[i] == ''):
            continue
        tweet.append(tweettemp[i])
    for i in range(0, len(tweet)):
        analiz_yap(tweet[i])



def grafik():
    global sayac_n
    global sayac_p
    label = ('Pozitif', 'Negatif')
    sizes = [sayac_p, sayac_n]
    color = ['green', 'red']
    plt.pie(sizes, labels=label, colors=color, autopct='%1.1f%%')
    plt.title('Konu hakkındaki düşünce oranı')
    plt.show()

pencere = Tk()
pencere.geometry("3000x3000+10+10")
pencere.title("Duygu Analizi")
pencere.configure(background="white")


isim = Label(pencere, text = "Aranacak konuyu giriniz:",font="Times 15",bg='white')
isim.place(x=420,y=70)


konu = Entry(pencere)
konu.place(x=425,y=100,width = 370)

isim2 = Label(pencere, text = "Türkçe veri için 1 İngilizce veri için 2 giriniz:",font="Times 15",bg='white')
isim2.place(x=420,y=120)

veri = Entry(pencere)
veri.place(x=425,y=150,width = 370)


buton = Button(pencere, text="Analiz Yap", command = tweetCek)
buton.place(x=525,y=175)

buton2 = Button(pencere,text="Grafiksel Göster",command = grafik)
buton2.place(x=600,y=175)

imgpos = ImageTk.PhotoImage(Image.open('if_face-smile_118880.png'))
labelpos = Label(pencere,image=imgpos)
labelpos.place(x=230,y=70)

imgneg = ImageTk.PhotoImage(Image.open('if_face-sad_118878.png'))
labelneg = Label(pencere,image=imgneg)
labelneg.place(x=850,y=70)


pencere.mainloop()