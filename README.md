# Implementasi Forward Propagation untuk Feed Forward Neural Network
Tugas Besar Machine Learning Bagian A

Anggota Kelompok:
- 13520001 - Fayza Nadia
- 13520014 - Muhammad Helmi Hibatullah
- 13520026 - Muhammad Fajar Ramadhan
- 13520029 - Muhammad Garebaldhie Er Rahman

## Implemented class and funcitons
1. 

## Library used
1. numpy
2. graphviz for visualization

## Model
Model yang dibuat menggunakan format seperti berikut

file `sigmoid.json`
```json
{
  "layers": 2,
  "activation_functions": ["sigmoid"],
  "neurons": [2, 3],
  "weights": [
    [
      [0.4, 0.2, 0.1],
      [0.2, 0.4, 0.2],
      [0.1, 0.2, 0.4]
    ]
  ],
  "rows": 1,
  "data_names": ["x1", "x2"],
  "data": [[0.2, 0.4]],
  "target_names": ["false", "true"],
  "target": [[0.617747, 0.58904, 0.574442]],
  "max_sse": 0.000001
}
```

`layers`: berisi banyaknya layer pada ffnn. Input layer merupakan layer sehingga perlu dimasukan juga ke dalam array neurons  
`activation_functions`: memiliki jumlah `layers - 1` karena activation functions menghubungkan dari layer ke layer. Activation functions yang valid berupa `sigmoid`, `relu`, `linear`, dan `softmax`  
`neurons`: berisi banyaknya neurons pada setiap layers  
`weights`: weights berisi bobot yang menghubungkan setiap layer. Weights harus memiliki panjang `layers - 1`.  
- dimensi 1 merupakan array yang menyimpan weights dari setiap layer
- dimensi 2 merupakan array yang menyimpan weights dari setiap neuron. Misal layer berikutnya ialah `y`, index ke 0 artinya bobot untuk `y1`, index 1 untuk `y2` dan index 2 untuk `y3` 
- dimensi 3 merupakan array yang menyimpan bobot dari suatu neuron dimulai dari bias. `[0.4, 0.2, 0.1]` berarti 0.4 adalah bias, 0.2 merupakan bobot `x1` dan 0.1 merupakan bobot `x2`  

`rows`: merupakan panjang array data yang diberikan  
`data_names`: merupakan nama attribut dari data yang diberikan  
`data`: merupakan input yang akan di prediksi  
`target_names`: merupakan hasil kelas prediksi yang menyatakan klasifikasi biner  
`target`: merupakan hasil output dari FFNN  
`max_sse`: merupakan max squared sum error yang ditoleransi 

## Visualization

## How to Run

### Local
1. Create virtualenv by using `virtualenv venv`
2. Activate virtualenv
   1. Windows: `./venv/Scripts/activate`
   2. Unix: `source ./venv/bin/activate`
3. Install all the dependencies `pip install -r requirements.txt`
4. Run the main program `python main.py`

### ipynb
1. Use `google colab`, `jupyter` or `jupyter for vscode` for opening ipynb files
2. Run all the celss 