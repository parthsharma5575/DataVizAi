document.addEventListener('DOMContentLoaded', function() {
    const authModal = new bootstrap.Modal(document.getElementById('authModal'));
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    const otpForm = document.getElementById('otp-form');
    const loginError = document.getElementById('login-error');
    const signupError = document.getElementById('signup-error');
    const otpError = document.getElementById('otp-error');
    const loginTabBtn = document.getElementById('login-tab');
    const signupTabBtn = document.getElementById('signup-tab');
    const otpTabBtn = document.getElementById('otp-tab');
    const otpTimer = document.getElementById('otp-timer');
    const resendOtpBtn = document.getElementById('resend-otp');
    let timerInterval;

    // Show modal if URL has ?login=true
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('login') === 'true') {
        authModal.show();
    }

    function startOtpTimer() {
        let duration = 600; // 10 minutes
        resendOtpBtn.disabled = true;
        timerInterval = setInterval(function() {
            const minutes = Math.floor(duration / 60);
            let seconds = duration % 60;
            seconds = seconds < 10 ? '0' + seconds : seconds;
            otpTimer.textContent = minutes + ':' + seconds;
            duration--;
            if (duration < 0) {
                clearInterval(timerInterval);
                otpTimer.textContent = '00:00';
                resendOtpBtn.disabled = false;
            }
        }, 1000);
    }

    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        loginError.classList.add('d-none');
        const email = document.getElementById('login-email').value;
        const password = document.getElementById('login-password').value;
        console.log('Logging in with:', email, password);

        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();
        console.log('Login response:', data);
        if (response.ok) {
            if (data.role === 'admin') {
                window.location.href = '/admin/dashboard';
            } else {
                window.location.href = '/upload';
            }
        } else {
            loginError.textContent = data.error;
            loginError.classList.remove('d-none');
        }
    });

    signupForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        signupError.classList.add('d-none');
        const name = document.getElementById('signup-name').value;
        const email = document.getElementById('signup-email').value;
        const phone = document.getElementById('signup-phone').value;
        const password = document.getElementById('signup-password').value;
        const confirmPassword = document.getElementById('signup-confirm-password').value;

        if (password !== confirmPassword) {
            signupError.textContent = "Passwords do not match.";
            signupError.classList.remove('d-none');
            return;
        }
        console.log('Signing up with:', name, email, phone);

        const response = await fetch('/auth/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name, email, phone, password })
        });

        const data = await response.json();
        console.log('Signup response:', data);
        if (response.ok) {
            document.getElementById('otp-user-id').value = data.userId;
            loginTabBtn.parentElement.classList.add('d-none');
signupTabBtn.parentElement.classList.add('d-none');
            otpTabBtn.parentElement.classList.remove('d-none');
            const otpTab = new bootstrap.Tab(otpTabBtn);
            otpTab.show();
            startOtpTimer();
        } else {
            signupError.textContent = data.error;
            signupError.classList.remove('d-none');
        }
    });

    otpForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        otpError.classList.add('d-none');
        const userId = document.getElementById('otp-user-id').value;
        const otp = document.getElementById('otp-code').value;
        console.log('Verifying OTP for user:', userId);

        const response = await fetch('/auth/verify-otp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ userId, otp })
        });

        const data = await response.json();
        console.log('OTP response:', data);
        if (response.ok) {
            window.location.href = '/upload';
        } else {
            otpError.textContent = data.error;
            otpError.classList.remove('d-none');
        }
    });

    resendOtpBtn.addEventListener('click', async function() {
        const userId = document.getElementById('otp-user-id').value;
        const response = await fetch('/auth/resend-otp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ userId })
        });
        const data = await response.json();
        if (response.ok) {
            startOtpTimer();
        } else {
            otpError.textContent = data.error;
            otpError.classList.remove('d-none');
        }
    });
});
