# Data Visualization for Features Comparison Using JavaScript Library D3.js

The HTML file uses the JavaScript library D3.js to dynamically visualize and compare the features and top-2 principle components (PCs) between two groups of patients. The interactive interfaces helps view and compare data in an easy and meaningful way. Interactive options include clicking dots directly, dragging slider, clicking buttons, choosing subgroup from a drop-down list. The visualization is a useful tool for feature engineering and/or selection.

## Visualization and Description

In the interface, top-2 PCs are shown as dots on the left to show a overview of the data set. When one dot is clicked, the normalized mean, one standard deviation, and top-10 features (normalized) corresponding to the dot will show on the right. To see the original values for each dot, just move the mouse over the dot. An information box will show those values for this highlighted dot.

To switch the left graph between all the patients (orange and blue dots) to the patients that were moved to ICU (green and red dots), we just choose the corresponding group from the drop-down list. All the dots in the left graph will be updated accordingly. This means that only one larger scatter plot on the left and two smaller top-10 feature scatter plots on the right are sufficient to present the data.

The snapshots for the dynamic interface can be found from the following poster.

![Dataviz_project_Poster_github](https://user-images.githubusercontent.com/42804316/57790170-32f6f580-7708-11e9-9aa0-2f12f3e7a5a8.png)
