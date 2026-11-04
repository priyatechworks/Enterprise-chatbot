from pathlib import Path
import fitz
out=Path(__file__).parent / "data" / "documents" / "sample_employee_policy.pdf"
out.parent.mkdir(parents=True,exist_ok=True)
content=[
("Employee Leave Policy", "Full-time employees receive 20 paid annual leave days per calendar year. Leave requests should be submitted in the HR portal at least five working days before the planned leave. A manager must approve each request. Unused annual leave may carry over up to five days into the next year; any amount above five days expires on December 31."),
("Sick Leave", "Employees receive 10 paid sick leave days per calendar year. If an employee is absent for more than three consecutive working days, they should provide a medical certificate to Human Resources. Sick leave does not carry over to the next year."),
("Expense Reimbursement", "Employees may claim approved business travel expenses within 30 days of the trip. Receipts are required for individual expenses over 25 dollars. A manager must approve the expense report before Finance processes reimbursement. Personal purchases are not reimbursable."),
]
doc=fitz.open()
for title,body in content:
    page=doc.new_page(); page.insert_text((72,72),title,fontsize=18); page.insert_textbox(fitz.Rect(72,110,520,740),body,fontsize=12)
doc.save(out)
print(out)
