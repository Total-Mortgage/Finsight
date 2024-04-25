select elpl.*, c.[Commission Expenses], elpl.[Back End Net] + elpl.[Front End Net] + elpl.[Loan Level Expenses] + elpl.[Loan Level Revenue] + c.[Commission Expenses] as [Total Profit] from(
select el.*, pl.[Loan Level Expenses], pl.[Loan Level Revenue] from(

SELECT 
Encompass_Loan_Number__c AS [Loan Number], 
Branch_Code__c AS [Branch Code], 
Loan_Officer__c AS [Loan Officer], 
Loan_Amount__c AS [Loan Amount], 
cast(Encompass_File_Started_Date__c as Date) AS [Started Date], 
cast(Pre_Approval_Initial_Print_Date__c as Date) AS [Pre Approval Date], 
case when Pre_Approval_Initial_Print_Date__c is not null then 1 else 0 end as [Has Pre Approval],
case when Pre_Approval_Initial_Print_Date__c is not null then Loan_Amount__c else 0 end as [Pre Approval Volume],
cast(Application_Taken_Date__c as Date) AS [Application Date], 
case when Application_Taken_Date__c is not null then 1 else 0 end as [Has Application],
case when Application_Taken_Date__c is not null then Loan_Amount__c else 0 end as [Application Volume],
cast(Lock_Date__c as Date) AS [Lock Date], 
case when Lock_Date__c is not null then 1 else 0 end as [Has Rate Lock],
case when Lock_Date__c is not null then Loan_Amount__c else 0 end as [Lock Volume],
cast(Closing_Completion_Date__c as Date) AS [Closing Date],
case when Closing_Completion_Date__c is not null then 1 else 0 end as [Is Closed],
case when Closing_Completion_Date__c is not null then Loan_Amount__c else 0 end as [Closed Volume],
Credit_Score__c AS [Credit Score],
DTI__c AS [DTI],
LTV__c AS [LTV],
Last_Completed_Milestone__c AS [Last Completed Milestone],
Underwriter_Name__c AS [Underwriter Name],
Loan_Processor__c AS [Loan Processor],
BUYERS_AGENT_CONTACT_NAME_VEND_X139 AS [Buyers Agent Name],
Sellers_Agent_Name__c AS [Sellers Agent Name],
Subject_Property_State__c AS [Subject Property State],
Subject_Property_City__c AS [Subject Property City],
Subject_Property_Type__c AS [Subject Property Type],
Subject_Property_Purchase_Price__c AS [Subject Property Purchase Price],
Subject_Property_Appraised_Value__c AS [Subject Property Appraised Value],
Down_Payment__c AS [Down Payment],
Intended_Property_Use__c AS [Intended Property Use],
Lien_Position__c AS [Lien Position],
Loan_Program__c AS [Loan Program],
Loan_Program_Type__c AS [Loan Program Type],
Loan_Term__c AS [Loan Term],
Loan_Type__c AS [Loan Type],
Rate_Type__c AS [Rate Type],
Loan_Purpose__c AS [Loan Purpose],
NOTERATE_3 AS [Note Rate],
cast(DENIAL_X69 as Date) AS [Denial Date],
CX_LEAD_DETAIL AS [Lead Detail],
CX_OUTSIDE_LEADSOURCE AS [Outside Lead Source],
Self_Employed__c AS [Self Employed],
cast(Rate_Lock_Expiration_Date__c as Date) AS [Rate Lock Expiration Date],
Annual_Income__c AS [Annual Income],
Monthly_Debts__c AS [Monthly Debts],
cast(Greenlight_Approval_Date__c as Date) AS [Greenlight Approval Date],
REFERRALNAME_1822 AS [Referral Name],
Primary_Borrower_Account__c as [Primary Account Id],
case when LOANINFOCHANNEL_2626 = 'Banked - Retail' and Closing_Completion_Date__c is not null then round(((FrontBpsTMS - FrontBpsBranch) * Loan_Amount__c) / 100, 2) else Null end as [Front End Net],
case when Closing_Completion_Date__c is not null then round((CX_CORP_HEDGE_ADJ_SPREAD * Loan_Amount__c) / 100, 2) else Null end as [Back End Net]

FROM 
EL_JoinedToEncompassHigh

where year(Encompass_File_Started_Date__c) >= 2020
) el

left join 

(
select e.Encompass_Loan_Number__c, e.[Loan Level Expenses], r.[Loan Level Revenue] from(

(select Encompass_Loan_Number__c, sum(p.Amount) as [Loan Level Expenses]

from EL_JoinedToEncompassHigh e

left join AMB_DW.dbo.PandL_Revenue_Expenses_March_2022_Itemized_Final p

on e.Encompass_Loan_Number__c = p.LOANNUMB
where type='Expenses'
group by Encompass_Loan_Number__c
) e

left join 

(
select Encompass_Loan_Number__c, sum(p.Amount) as [Loan Level Revenue]

from EL_JoinedToEncompassHigh e

left join AMB_DW.dbo.PandL_Revenue_Expenses_March_2022_Itemized_Final p

on e.Encompass_Loan_Number__c = p.LOANNUMB
where type='Revenue'
group by Encompass_Loan_Number__c
) r

on e.Encompass_Loan_Number__c = r.Encompass_Loan_Number__c
)
) pl 

on el.[Loan Number] = pl.Encompass_Loan_Number__c
) elpl

left join 

(
select LOANNUMBER, sum(amount) * -1 as [Commission Expenses]

from Coldstorage.dbo.Compensafe_CombinedBonus_Commission

group by LOANNUMBER
) c

on elpl.[Loan Number] = c.LOANNUMBER