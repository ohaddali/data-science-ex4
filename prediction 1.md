
# Prediction 1 - Using Naive Bayes

### Naive Bayes classifier is probabilistic calssifier based on applying Bayes' theorem with strong independence assumptions between the features.
In this document we use the Gaussian naive Bayes classifier.

With this algorithm we got the best result that placed us in :

![leaderBoard](./Images/leaderBoard.PNG)


##Pre Processing the data

Installing the needed packages.


```python
%pylab inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
```

    Populating the interactive namespace from numpy and matplotlib
    

Reading the data into dataframe.


```python
df = pd.read_csv("./data/train.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
      <th>Loan_Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001002</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5849</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001003</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>1508.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001005</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3000</td>
      <td>0.0</td>
      <td>66.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001006</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2583</td>
      <td>2358.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001008</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6000</td>
      <td>0.0</td>
      <td>141.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LP001011</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>5417</td>
      <td>4196.0</td>
      <td>267.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LP001013</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2333</td>
      <td>1516.0</td>
      <td>95.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LP001014</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3036</td>
      <td>2504.0</td>
      <td>158.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LP001018</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4006</td>
      <td>1526.0</td>
      <td>168.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LP001020</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>12841</td>
      <td>10968.0</td>
      <td>349.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LP001024</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3200</td>
      <td>700.0</td>
      <td>70.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LP001027</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>2500</td>
      <td>1840.0</td>
      <td>109.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LP001028</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3073</td>
      <td>8106.0</td>
      <td>200.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LP001029</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1853</td>
      <td>2840.0</td>
      <td>114.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LP001030</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1299</td>
      <td>1086.0</td>
      <td>17.0</td>
      <td>120.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LP001032</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4950</td>
      <td>0.0</td>
      <td>125.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LP001034</td>
      <td>Male</td>
      <td>No</td>
      <td>1</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3596</td>
      <td>0.0</td>
      <td>100.0</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LP001036</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3510</td>
      <td>0.0</td>
      <td>76.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Urban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LP001038</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>4887</td>
      <td>0.0</td>
      <td>133.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LP001041</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>2600</td>
      <td>3500.0</td>
      <td>115.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LP001043</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>7660</td>
      <td>0.0</td>
      <td>104.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Urban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LP001046</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5955</td>
      <td>5625.0</td>
      <td>315.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LP001047</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2600</td>
      <td>1911.0</td>
      <td>116.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LP001050</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3365</td>
      <td>1917.0</td>
      <td>112.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LP001052</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>3717</td>
      <td>2925.0</td>
      <td>151.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LP001066</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>9560</td>
      <td>0.0</td>
      <td>191.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LP001068</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2799</td>
      <td>2253.0</td>
      <td>122.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LP001073</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>4226</td>
      <td>1040.0</td>
      <td>110.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LP001086</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>1442</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LP001087</td>
      <td>Female</td>
      <td>No</td>
      <td>2</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>3750</td>
      <td>2083.0</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>584</th>
      <td>LP002911</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2787</td>
      <td>1917.0</td>
      <td>146.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>585</th>
      <td>LP002912</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4283</td>
      <td>3000.0</td>
      <td>172.0</td>
      <td>84.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>586</th>
      <td>LP002916</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2297</td>
      <td>1522.0</td>
      <td>104.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>587</th>
      <td>LP002917</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2165</td>
      <td>0.0</td>
      <td>70.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>588</th>
      <td>LP002925</td>
      <td>NaN</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4750</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>589</th>
      <td>LP002926</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>2726</td>
      <td>0.0</td>
      <td>106.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>590</th>
      <td>LP002928</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3000</td>
      <td>3416.0</td>
      <td>56.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>591</th>
      <td>LP002931</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>6000</td>
      <td>0.0</td>
      <td>205.0</td>
      <td>240.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>592</th>
      <td>LP002933</td>
      <td>NaN</td>
      <td>No</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>9357</td>
      <td>0.0</td>
      <td>292.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>593</th>
      <td>LP002936</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3859</td>
      <td>3300.0</td>
      <td>142.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>594</th>
      <td>LP002938</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>16120</td>
      <td>0.0</td>
      <td>260.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>595</th>
      <td>LP002940</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3833</td>
      <td>0.0</td>
      <td>110.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>596</th>
      <td>LP002941</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>Yes</td>
      <td>6383</td>
      <td>1000.0</td>
      <td>187.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>N</td>
    </tr>
    <tr>
      <th>597</th>
      <td>LP002943</td>
      <td>Male</td>
      <td>No</td>
      <td>NaN</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2987</td>
      <td>0.0</td>
      <td>88.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>598</th>
      <td>LP002945</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>9963</td>
      <td>0.0</td>
      <td>180.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>599</th>
      <td>LP002948</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5780</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>600</th>
      <td>LP002949</td>
      <td>Female</td>
      <td>No</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>416</td>
      <td>41667.0</td>
      <td>350.0</td>
      <td>180.0</td>
      <td>NaN</td>
      <td>Urban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>601</th>
      <td>LP002950</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>NaN</td>
      <td>2894</td>
      <td>2792.0</td>
      <td>155.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>602</th>
      <td>LP002953</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5703</td>
      <td>0.0</td>
      <td>128.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>603</th>
      <td>LP002958</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3676</td>
      <td>4301.0</td>
      <td>172.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>604</th>
      <td>LP002959</td>
      <td>Female</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>12000</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>605</th>
      <td>LP002960</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2400</td>
      <td>3800.0</td>
      <td>NaN</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>N</td>
    </tr>
    <tr>
      <th>606</th>
      <td>LP002961</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3400</td>
      <td>2500.0</td>
      <td>173.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>607</th>
      <td>LP002964</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3987</td>
      <td>1411.0</td>
      <td>157.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>608</th>
      <td>LP002974</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3232</td>
      <td>1950.0</td>
      <td>108.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>609</th>
      <td>LP002978</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2900</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>610</th>
      <td>LP002979</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4106</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Rural</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>611</th>
      <td>LP002983</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>8072</td>
      <td>240.0</td>
      <td>253.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>612</th>
      <td>LP002984</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>7583</td>
      <td>0.0</td>
      <td>187.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>613</th>
      <td>LP002990</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>4583</td>
      <td>0.0</td>
      <td>133.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>614 rows × 13 columns</p>
</div>



### Remove Nulls

To impute all the missing values in the data frame we will use the fillna method using information from HW4 file.

First we need to impute the missing values from the Self_Employed column, in order to use it to create a Pivot Table to fill the nulls from LoanAmount column.



```python
df['Self_Employed'].fillna('No',inplace=True)

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
```

We will impute the missing values for the rest of the columns using previous information from HW4 file. 


```python

df['Credit_History'].fillna(1, inplace=True)
df['Loan_Amount_Term'].fillna(360,inplace=True)
df['Dependents'].fillna(0, inplace=True)
df['Married'].fillna('Yes', inplace=True)
df['Gender'].fillna('Male', inplace=True)
```

Let's check if there is any nulls in the dataframe.


```python
df.apply(lambda x : sum(x.isnull()),axis=0)
```




    Loan_ID              0
    Gender               0
    Married              0
    Dependents           0
    Education            0
    Self_Employed        0
    ApplicantIncome      0
    CoapplicantIncome    0
    LoanAmount           0
    Loan_Amount_Term     0
    Credit_History       0
    Property_Area        0
    Loan_Status          0
    dtype: int64



Excellent!

## Building a Predictive Model in Python

Now we will build the Naive Bayes model on our data set.
Like in the example we will use the Skicit-Learn library, that requires from us to convert all inputs to numeric.

Next, we will import the required modules. 


```python
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cross_validation import KFold
```

Convert all the categorival variables into numeric


```python
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i].astype(str))
```

Now let's train the model on our data set. We will use only those variables : LoanAmount , Education , Property_Area , Credit_History



```python
nbModel = GaussianNB()
predictor_vars = ['LoanAmount','Education','Property_Area','Credit_History']
outcome_var = ['Loan_Status']
nbModel.fit(df[predictor_vars],df[outcome_var])

predictions = nbModel.predict(df[predictor_vars])
```

The accuracy of this training:


```python
print("Accuracy : %s" % "{0:.3%}".format(metrics.accuracy_score(predictions,df[outcome_var])))
```

    Accuracy : 80.945%
    

Perform k-fold CV with 5 folds


```python
kf = KFold(df.shape[0], n_folds=5)
error = [] 
for train ,test  in kf:
    train_predictors = (df[predictor_vars].iloc[train,:])
    train_target = df[outcome_var].iloc[train]
    
    # Training the algorithm using the predictors and target.
    nbModel.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(nbModel.score(df[predictor_vars].iloc[test,:], df[outcome_var].iloc[test]))
    
print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
nbModel.fit(df[predictor_vars],df[outcome_var])
```

    Cross-Validation Score : 80.457%
    




    GaussianNB(priors=None)



### The test data set
Now let's load the test data_set : 


```python
df_test = pd.read_csv("./data/test.csv")
df_test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Loan_ID</th>
      <th>Gender</th>
      <th>Married</th>
      <th>Dependents</th>
      <th>Education</th>
      <th>Self_Employed</th>
      <th>ApplicantIncome</th>
      <th>CoapplicantIncome</th>
      <th>LoanAmount</th>
      <th>Loan_Amount_Term</th>
      <th>Credit_History</th>
      <th>Property_Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LP001015</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5720</td>
      <td>0</td>
      <td>110.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LP001022</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3076</td>
      <td>1500</td>
      <td>126.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LP001031</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5000</td>
      <td>1800</td>
      <td>208.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LP001035</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2340</td>
      <td>2546</td>
      <td>100.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LP001051</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3276</td>
      <td>0</td>
      <td>78.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LP001054</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>Yes</td>
      <td>2165</td>
      <td>3422</td>
      <td>152.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LP001055</td>
      <td>Female</td>
      <td>No</td>
      <td>1</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2226</td>
      <td>0</td>
      <td>59.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LP001056</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3881</td>
      <td>0</td>
      <td>147.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LP001059</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>13633</td>
      <td>0</td>
      <td>280.0</td>
      <td>240.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LP001067</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2400</td>
      <td>2400</td>
      <td>123.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LP001078</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3091</td>
      <td>0</td>
      <td>90.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LP001082</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>2185</td>
      <td>1516</td>
      <td>162.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LP001083</td>
      <td>Male</td>
      <td>No</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4166</td>
      <td>0</td>
      <td>40.0</td>
      <td>180.0</td>
      <td>NaN</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LP001094</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>NaN</td>
      <td>12173</td>
      <td>0</td>
      <td>166.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LP001096</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4666</td>
      <td>0</td>
      <td>124.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LP001099</td>
      <td>Male</td>
      <td>No</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5667</td>
      <td>0</td>
      <td>131.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LP001105</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4583</td>
      <td>2916</td>
      <td>200.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LP001107</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3786</td>
      <td>333</td>
      <td>126.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LP001108</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>9226</td>
      <td>7916</td>
      <td>300.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LP001115</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>1300</td>
      <td>3470</td>
      <td>100.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LP001121</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>1888</td>
      <td>1620</td>
      <td>48.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LP001124</td>
      <td>Female</td>
      <td>No</td>
      <td>3+</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2083</td>
      <td>0</td>
      <td>28.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LP001128</td>
      <td>NaN</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3909</td>
      <td>0</td>
      <td>101.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>23</th>
      <td>LP001135</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3765</td>
      <td>0</td>
      <td>125.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LP001149</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5400</td>
      <td>4380</td>
      <td>290.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>25</th>
      <td>LP001153</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>0</td>
      <td>24000</td>
      <td>148.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LP001163</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4363</td>
      <td>1250</td>
      <td>140.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LP001169</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>7500</td>
      <td>3750</td>
      <td>275.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>28</th>
      <td>LP001174</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3772</td>
      <td>833</td>
      <td>57.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>29</th>
      <td>LP001176</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2942</td>
      <td>2382</td>
      <td>125.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>337</th>
      <td>LP002856</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2292</td>
      <td>1558</td>
      <td>119.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>338</th>
      <td>LP002857</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>2360</td>
      <td>3355</td>
      <td>87.0</td>
      <td>240.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>339</th>
      <td>LP002858</td>
      <td>Female</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4333</td>
      <td>2333</td>
      <td>162.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>340</th>
      <td>LP002860</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>2623</td>
      <td>4831</td>
      <td>122.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>341</th>
      <td>LP002867</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>3972</td>
      <td>4275</td>
      <td>187.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>342</th>
      <td>LP002869</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3522</td>
      <td>0</td>
      <td>81.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>343</th>
      <td>LP002870</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4700</td>
      <td>0</td>
      <td>80.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>344</th>
      <td>LP002876</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>6858</td>
      <td>0</td>
      <td>176.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>345</th>
      <td>LP002878</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>8334</td>
      <td>0</td>
      <td>260.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>346</th>
      <td>LP002879</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3391</td>
      <td>1966</td>
      <td>133.0</td>
      <td>360.0</td>
      <td>0.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>347</th>
      <td>LP002885</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>2868</td>
      <td>0</td>
      <td>70.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>348</th>
      <td>LP002890</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3418</td>
      <td>1380</td>
      <td>135.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>349</th>
      <td>LP002891</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>2500</td>
      <td>296</td>
      <td>137.0</td>
      <td>300.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>350</th>
      <td>LP002899</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Graduate</td>
      <td>No</td>
      <td>8667</td>
      <td>0</td>
      <td>254.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>351</th>
      <td>LP002901</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2283</td>
      <td>15000</td>
      <td>106.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>352</th>
      <td>LP002907</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5817</td>
      <td>910</td>
      <td>109.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>353</th>
      <td>LP002920</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5119</td>
      <td>3769</td>
      <td>120.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>354</th>
      <td>LP002921</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>5316</td>
      <td>187</td>
      <td>158.0</td>
      <td>180.0</td>
      <td>0.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>355</th>
      <td>LP002932</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Graduate</td>
      <td>No</td>
      <td>7603</td>
      <td>1213</td>
      <td>197.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>356</th>
      <td>LP002935</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3791</td>
      <td>1936</td>
      <td>85.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>357</th>
      <td>LP002952</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2500</td>
      <td>0</td>
      <td>60.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>358</th>
      <td>LP002954</td>
      <td>Male</td>
      <td>Yes</td>
      <td>2</td>
      <td>Not Graduate</td>
      <td>No</td>
      <td>3132</td>
      <td>0</td>
      <td>76.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>359</th>
      <td>LP002962</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4000</td>
      <td>2667</td>
      <td>152.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>360</th>
      <td>LP002965</td>
      <td>Female</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>8550</td>
      <td>4255</td>
      <td>96.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>361</th>
      <td>LP002969</td>
      <td>Male</td>
      <td>Yes</td>
      <td>1</td>
      <td>Graduate</td>
      <td>No</td>
      <td>2269</td>
      <td>2167</td>
      <td>99.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>362</th>
      <td>LP002971</td>
      <td>Male</td>
      <td>Yes</td>
      <td>3+</td>
      <td>Not Graduate</td>
      <td>Yes</td>
      <td>4009</td>
      <td>1777</td>
      <td>113.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>363</th>
      <td>LP002975</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>4158</td>
      <td>709</td>
      <td>115.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Urban</td>
    </tr>
    <tr>
      <th>364</th>
      <td>LP002980</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>3250</td>
      <td>1993</td>
      <td>126.0</td>
      <td>360.0</td>
      <td>NaN</td>
      <td>Semiurban</td>
    </tr>
    <tr>
      <th>365</th>
      <td>LP002986</td>
      <td>Male</td>
      <td>Yes</td>
      <td>0</td>
      <td>Graduate</td>
      <td>No</td>
      <td>5000</td>
      <td>2393</td>
      <td>158.0</td>
      <td>360.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
    <tr>
      <th>366</th>
      <td>LP002989</td>
      <td>Male</td>
      <td>No</td>
      <td>0</td>
      <td>Graduate</td>
      <td>Yes</td>
      <td>9200</td>
      <td>0</td>
      <td>98.0</td>
      <td>180.0</td>
      <td>1.0</td>
      <td>Rural</td>
    </tr>
  </tbody>
</table>
<p>367 rows × 12 columns</p>
</div>



We need to pre-process the data in the same way we used on the train set.


```python
df_test['Self_Employed'].fillna('No',inplace=True)
df_test['LoanAmount'].fillna(df_test[df_test['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
df_test['Credit_History'].fillna(1, inplace=True)
df_test['Loan_Amount_Term'].fillna(360,inplace=True)
df_test['Dependents'].fillna(0, inplace=True)
df_test['Married'].fillna('Yes', inplace=True)
df_test['Gender'].fillna('Male', inplace=True)
```

And convert the columns to numeric


```python
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    df_test[i] = le.fit_transform(df_test[i].astype(str))
```

We will predict only the variables we chose before :LoanAmount , Education , Property_Area , Credit_History


```python
loan_ids = df_test['Loan_ID']
df_test = df_test[predictor_vars]
result = nbModel.predict(df_test)
result
```




    array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
           0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)



This result included the nunmbers 1 and 0 , but the expected result is 'Y' and 'N'.


```python
result  = ['Y' if res==1 else 'N' for res in result]
```

Now only remains to write the results to a file and submit it


```python
resultsToWrite = pd.DataFrame({'Loan_ID' : loan_ids, 'Loan_Status' : result})
resultsToWrite.to_csv("NaiveBayes_submission.csv")
```

## The prediction Score
![NaiveBayes prediction Score](./Images/NaiveBayes_score.PNG)
