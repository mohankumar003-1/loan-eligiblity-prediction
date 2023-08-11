from django.db import models
from django.conf import settings

# Create your models here.
class approvals(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    author = models.CharField(max_length=100, default='Default Author')

    GENDER_CHOICES = (
        ('1', 'Male'),
        ('0', 'Female')
    )
    MARRIED_CHOICES = (
        ('1', 'Yes'),
        ('0', 'No')
    )
    GRADUATED_CHOICES = (
        ('1', 'Graduated'),
        ('0', 'Not_Graduated')
    )
    SELFEMPLOYED_CHOICES = (
        ('1', 'Yes'),
        ('0', 'No')
    )
    PROPERTY_CHOICES = (
        ('0', 'Rural'),
        ('1', 'Semiurban'),
        ('2', 'Urban')
    )



    Dependents = models.IntegerField(default=0, null=True)
    ApplicantIncome = models.IntegerField(default=0, null=True)
    CoapplicantIncome  = models.IntegerField(default=0, null=True)
    LoanAmount = models.IntegerField(default=0, null=True)
    Loan_Amount_Term = models.IntegerField(default=0, null=True)
    Credit_History = models.IntegerField(default=0, null=True)
    Gender = models.IntegerField( choices = GENDER_CHOICES)
    Married = models.IntegerField( choices = MARRIED_CHOICES)
    Education = models.IntegerField( choices = GRADUATED_CHOICES)
    Self_Employed = models.IntegerField( choices = SELFEMPLOYED_CHOICES)
    Property_Area = models.IntegerField( choices = PROPERTY_CHOICES)
   
# Create your models here.
