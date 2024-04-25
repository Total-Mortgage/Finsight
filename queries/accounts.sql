select distinct id as [Account Id],
Lead_attributed_to__c as [Attributed Lead Id]

from Account
where Lead_attributed_to__c is not null