document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    form.addEventListener('submit', function(e) {
        const visaType = document.getElementById('visa_type').value;
        const country = document.getElementById('country').value;
        const urgency = document.getElementById('urgency').value;
        
        if (!visaType || !country || !urgency) {
            alert('All fields are required!');
            e.preventDefault();
        } else {
            // Show progress bar
            document.querySelector('.progress').style.display = 'block';
        }
    });
    
    // Tooltips for dropdowns
    const selects = document.querySelectorAll('select');
    selects.forEach(select => {
        select.addEventListener('mouseover', function() {
            this.title = 'Select an option to proceed.';
        });
    });
});