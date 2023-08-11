from django.shortcuts import render

import pandas as pd
from .models import approvals
from .code import make_predictions
def loan_pred(request):
    if request.method =='POST':
            #firstname = request.POST.get['firstname']
            #lastname = request.POST.get['lastname']
      Dependents = request.POST.get('Dependents')
      ApplicantIncome = request.POST.get('ApplicantIncome')
      CoapplicantIncome = request.POST.get('CoapplicantIncome')
      LoanAmount = request.POST.get('LoanAmount')
      Loan_Amount_Term = request.POST.get('Loan_Amount_Term')
      Credit_History = request.POST.get('Credit_History')
      Gender = request.POST.get('Gender')
      Married = request.POST.get('Married')
      Education = request.POST.get('Education')
      Self_Employed = request.POST.get('Self_Employed')
      Property_Area = request.POST.get('Property_Area')
      print(Married)
      data = approvals(Gender = Gender, Married = Married, Dependents = Dependents, Education = Education, Self_Employed = Self_Employed, ApplicantIncome = ApplicantIncome, CoapplicantIncome = CoapplicantIncome, LoanAmount = LoanAmount, Loan_Amount_Term  = Loan_Amount_Term, Credit_History = Credit_History, Property_Area = Property_Area)
            # print(firstname, lastname, dependents,married, area)
      data.save()
      a = make_predictions(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, int(Property_Area))
      context = {}
      context['a'] = a
      return render(request,'base.html',context)  
    return render(request,'index.html')
      
 
    

# Create your views here.
