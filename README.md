# Kratos

Instructions for running on PSU machines.  

Clone Kratos Repo:  
    git clone https://github.com/ZackSalah/Kratos.git  
    cd Kratos  
Checkout remory_attributes branch:  
	git fetch origin  
	git checkout -b remory_attributes origin/remory_attributes  

Start kinit session and SSH to a GPU server  

Install pandas:  
	  pip install pandas  
Install sklearn package:  
	  pip install scikit-learn  
Run training:  
	  python attributeML.py
