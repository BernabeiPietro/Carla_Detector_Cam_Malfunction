# carla-detector-cam-malfunction

## Istruzioni per avviare CARLA:
1) copia il file client_example.py del seguente progetto nella cartella PythonClient all'interno della directory Carla 0.8.4
2) crea installa tutti i pacchetti necessari all'avvio
  * è presente all'interno del progetto un file requirements.txt  per l'installazione dei pacchetti necessari
3) avviare il lato server con la seguente dicitura:
DISPLAY=:21 ./CarlaUE4.sh -opengl -world-port=10500 -quality=low
4) avviare lato client con la seguene dicitura:
python3 client_example.py --images-to-disk --location=_out -p=10500
### importante il file client_example.pyè impostato per acquisire 500 episodi. Per cambiare le impostazioni si consiglia di modificare direttamente il codice.

## Istruzioni per la creazione dei dataset:
### Al momento i fallimenti sviluppati sono: "blur", "black", "brightness",  "200_death_pixels","nodemos","noise","sharpness","brokenlens","icelens","banding","50_death_pixels","greyscale","condensation","dirty_lens","chromaticaberration","rain". Per ulteriori aggiunte o modifiche il programma che è da modificare è modify_photo.py. Si consiglia al momento dell'aggiunta di un nuovo fallimento di definire l'intestazione della funzione che verrà richiamata in maniera identica a quelle precedenti. Esempio:
def blur (images,progressiveName,pathModified
### un altra cosa da fare al momento della definizione di un nuovo fallimento per essere richiamato in maniera automatica dal sistema di inserire nell'oggetto a fine file, dispatcher, il nome della funzione e una stringa identificativa. Stringa identificativa che dovrà essere aggiunta anche in carla_image_modify.py e tensor.py. Più precisamente all'iterno della lista classes_of_modified di tutti e due i file.

### Avvio:
1) Nel file carla_image_modify.py  nella funzione manage_image le prime quattro righe indicano il numero di episodi che devono essere impiegate per ciascun set. Modificare al fine di cambiare il numero di episodi per ciascun set.  Si ricorda che il programma non usa lo stesso  episodio per due set diversi dello stesso fallimento.
2) In fondo allo stesso file sono presenti tre elementi path, classes_of_modified,  mp, e la funzione manage_image().
  * In path  deve essere indicato il percorso in cui sono stati generati gli episodi di CARLA; di solito nel seguente percorso .../carla0.8.4/PythonClient/_out/ . All'itnerno della stessa directory saranno salvati anche i dataset
  * classes_of_modified contiene i nomi di tutti i fallimenti sviluppati, oltre a  servire per richiamare le funzione e gestire i nomi dei vari dataset
  * manager_of_path è un programma sviluppato presente nella reposity che gestisce la creazione e strutturazione di tutti i dataset. IMPORTANTE l'ultimo dei parametri è un valore booleano che a seconda di come viene impostato definisce che tipo  di dataset si vuole generare:
    ** True crea dataset mono fallimento
    ** False crea un unico dataset multifallimento- sotto la dicitura All
  * manage_image: 
      **  mp è il manager dei percorsi
      ** classes_of_modified indica che fallimenti si vogliono andare a modificare. Nel caso in cui  si voglia lavorare con un numero limitato di fallimenti, si consiglia di modficare la dicitura precedente con classes_of_modified[#indA,#indB] o comunque in modo tale da passare una lista al programma.
      ** l'ultimo parametro è un valore booleano che indica se si vuole popolare il test set asimmetrico (True) oppure no (False). Valido solo per i dataset monoguasti.
 3) avviare semplicemte il programma con: 
 python3 carla_image_modify.py
 
 # Istruzioni per il training e testing della rete.
tensor.py è il programma dedicato alla training e testing della rete.

