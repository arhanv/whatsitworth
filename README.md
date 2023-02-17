# What's it Worth?: Exploring the Housing Market around Chicago
Multiple logistic regression analysis of housing prices in the Chicago metropolitan area

## Background
This was a project I completed for my [Principles of Data Science](https://www.ds100.org) class at Berkeley as a sophomore. The dataset I used is publicly available from the website of the [Cook County Assessor's Office](https://www.cookcountyassessor.com/community-data). The overarching goal here was to develop my understanding of the feature engineering and linear modelling process by investigating the various ways in which I could incrementally improve the validation and training accuracy of a model derived from a feature-dense dataset, but also considering the loaded socioeconomic implications that arise from looking at indicators that are deeply intertwined with the concepts of red-lining, segregation, and gentrification.

Here are some of the questions related to causality, modelling and inference that I explored through this study:
* How can I balance underfitting and overfitting concerns in a model with so many (seemingly) independent features?
* Can causal correlations amongst different features pose a multicollinearity problem in my regression analysis?
* How do the types of pricing models chosen by tax authorities impact assessment outcomes for individuals?
* Do the pricing patterns predicted by my model seem to mirror the historical inequalities experienced by minority residents in Cook County?
  * Does this mean that our assessment practices are only increasing the severity of damage caused by historical injustices?
 
## Dimensionality of Data
| Dataset | Dimensions | 
|-----:|---------------|
|   `training_set`|   (204792, 62)    |
|     `testing_set`|  (68264, 61)     |

## Feature Spec
There were 61 + 1 features in the training data, which is mostly redundant to include here. Here's a few examples of what they looked like:
| Feature | Data Type | Description |
| ------: | --------- | ----------- |
| `PIN` | `str` | Unique Permanent Identification Number for each property |
| `floodplain` | `int` | Binary indicator - whether property is close to a floodplain |
| `land_square_feet` | `float` | Area of property measured in square feet |
| `ohare_noise` | `int` | Binary indicator - whether property is within 1/4 mile of airport runway |
| `central_heating` | `int` | Central heating type - 1 = Warm air, 2 = Hot water steam, 3 = Electric, 4 = Other |
| + 57 more| ... | ... |

## Model Design
The logistic regression model I ended up with through multiple rounds of k-cross-fold validation operated as follows:

| Feature Name (Predictor)                    | Transformations        |
|---------------------------------|------------------------|
| $\log$ Building Square Footage         | `log` transform        |
| Indicator - Architect-Designed  | -                |
| Age (in decades)                | `//10` floor transform |
| Indicator - Garage              | -                    |
| Apartments                      | -                   |
| O'Hare Noise                    | -                   |
| Indicator - Porch               | -                   |
| Road Proximity                  | -                   |
| Neighborhood Code               | `one_hot_encode`       |
| Property Class                  | `one_hot_encode`       |
| Town Code                       | `one_hot_encode`       |


| Feature Name (Response) | Transformation |
|------------|----------------|
| $\log$ Sale Price | `log` transform |

The logistic regression equation resembles the form $\log (\text{sale price}) = \alpha \cdot \text{sqft} + \beta \cdot \mathbb{I}[\text{architect}] + \gamma \cdot \frac{\text{age}}{10} + ...$

## Modelling Results

### Training Error 
Following hyperparameter tuning and additional feature tweaking, the `RMSE` of predictions on the training set was `151450.53`.

### Test Error
The `RMSE` of my predictions in private testing conducted by the course instructors was `169918.63`.

This indicates that the model was not significantly overfitted. The `RMSE` threshold for the highest grade in the class was 200k, which I had surpassed.

## Inferences

### A Note About Accuracy
I optimized the predictive accuracy of my model by trying a few different models based on features such as the design plan of the home and whether or not a garage or porch is included, but they decreased the RMSE of the model by a very small amount. What I found was most helpful in building an accurate model was including the locality information, i.e. **neighborhood and town codes** in one-hot-encoded columns. What seems to have the most impact on the pricing of a house is the neighborhood in which it is located. I also tried using previous sale information like land and building estimates, although that did not work and vastly increased my error value. I think that my final model was largely informed by the features that I intuitively felt could impact the desirability of a house (proximity to the airport and roads, square footage, age) as well as the neighborhood information that seemed to (literally) exponentially increase the model's accuracy upon its addition.

### A Note About Conventional Assessment Systems
The error value of any given model's predictions is representative of its inaccuracy in "correctly" estimating the value of a home, so by extension, it is also a measure of a unfairness of the model in assigning property values - especially if there are systemic discrepancies in the process as there were in Cook County. If an erring system is used to estimate the values of properties in the context of taxation, this could result in certain homeowners having to pay excessive taxes on their homes - so if the error results in "regressivity" (low values are overestimated while high values are underestimated) then the taxation imposed as a consequence of these estimates would be systemically unfair to those who own cheaper homes. In an economic context where purchasing a home is already virtually impossible for low-income families and the distribution of wealth is at an all-time-low, this would only perpetuate the presence of poverty in families who are already suffering. Certainly, some homeowners could benefit from having to pay lower taxes on their homes, but it would be in the best interest of the county to try to correct any systemic issues in taxation structures because they want to maximize their revenue - there is no point in asking low-income individuals for more taxes if high-income assets that are undertaxed could provide far more revenue. As such, accurate estimates would be in the best interests of homeowners as well as tax authorities - it would be simply unfair to purchase a home and then have to pay taxes on the property that you had no tangible method of estimating before making the purchase.
