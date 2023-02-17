# Kaggle-Competion
## Table of contents
* [About info](#about-info)
* [Requirements](#requirements-met)
* [Technologies](#technologies)
* [Status](#status)

## Cuprins
* [Preprocesare](#preprocesare)
* [Model](#model)

## Preprocesare

Preprocesarea datelor este un pas esential in Inteligenta Artificiala, intrucat clasificatorii nostrii se pot antrena doar cu date numerice, ci nu cu date brute (in cazul nostru cele 41570 de texte de antrenare).

Datele de antrenare brute reprezinta textele traduse din cele 3 dialecte englezesti (scotian, irlandez, britanic) in 5 limbi distincte: italiana, spaniola, olandeza, daneza si germana. Pentru a usura procesul de predictie a dialectelor initiale, textele au fost traduse folosind API-ul google inapoi in limba originara, engleza.

Pentru a transforma textele traduse in date numerice am folosit reprezentarea Bag-of-Words, unde celor mai frecvente n cuvinte din fiecare dialect le este asociat cate un numar, reprezentand frecventa cu care respectivul cuvant se regaseste la nivelul fiecarui text.

Pentru a obtine aceste n cele mai frecvente cuvinte din textele noastre este necesara o tokenizare a datelor, ce consta in: extragerea cuvintelor din fiecare text, lower-casing (transformarea majusculelor in minuscule), eliminarea stopwords (cuvinte lipsite de relevanta; ex: prepozitii, cuvinte de legatura etc.), cat si lemmatization (aducerea cuvintelor la forma lor de baza; ex: runs, running, ran – run).

## Model

In ceea ce priveste procesul de antrenare, modelul folosit a fost o retea neuronala feedforward, cunoscuta si sub denumirea de multilayer perceptron, intrucat consta in interconectarea mai multor straturi (layers) de perceptroni. 

* Caracteristici

Caracteristicile modelului sunt definite in urma preprocesarii si reprezinta reuniunea dictionarelor fiecarui dialect (dictionar = cele mai frecvente 1000 de cuvinte dintr-un dialect), unde eventualele duplicate optinute din reuniunea dictionarelor au fost eliminate. 

* Parametri

In cazul unei retele neuronale, parametrii sunt reprezentati de ponderi (weights), deoarece acestea sunt invatate de model pe parcursul procesului de antrenare. Pentru inceput, ponderile au fost initializate cu numere aleatoare (random). 

* Hiperparametri

Hiperparametrii unei retele neuronale sunt reprezentati de:
* numarul de epoci (20)
* rata de invatare (0.001) 
* batch-size (64)
* arhitectura modelului (1 strat de input – 1197 perceptroni, 1 strat hidden – 60 perceptroni, 1 strat de output – 3 perceptroni).
Rata de invatare, numarul de epoci, batch-size-ul si arhitectura modelului au fost alese pur experimental, pentru a incerca evitarea overfitting-ului (numar de epoci cat mai mic, rata de invatare cat mai mica, batch-size cat mai mic si arhitectura cat mai simplista).

* Antrenarea parametrilor

Procesul de antrenare are urmatorii pasi esentiali:

* forward propagation / forward pass: procesul de actualizare a ponderilor, avansand de la stratul de input catre cel de output.
* backward propagation / backward pass: consta in procesul de gradient descent, adica incercarea algoritmului de a minimiza functia de loss.

* Timpul de antrenare

Antrenarea dureaza aproximativ 16.8 secunde.

* Performanta pe Kaggle

Pentru cele 40% de date din setul de date de test public am obtinut 69.426% acuratete.
