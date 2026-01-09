document.addEventListener('DOMContentLoaded', function() {
    const usersTableBody = document.getElementById('users-table-body');
    const filesTableBody = document.getElementById('files-table-body');
    const filesPagination = document.getElementById('files-pagination');

    async function fetchUsers() {
        const response = await fetch('/admin/users');
        const users = await response.json();
        usersTableBody.innerHTML = '';
        users.forEach(user => {
            const row = `
                <tr>
                    <td>${user.id}</td>
                    <td>${user.name}</td>
                    <td>${user.email}</td>
                    <td>${user.phone || ''}</td>
                    <td>${user.role}</td>
                    <td>${user.status}</td>
                    <td>${new Date(user.created_at).toLocaleString()}</td>
                </tr>
            `;
            usersTableBody.innerHTML += row;
        });
    }

    async function fetchFiles(page = 1) {
        const response = await fetch(`/admin/files?page=${page}`);
        const data = await response.json();
        filesTableBody.innerHTML = '';
        data.files.forEach(file => {
            const row = `
                <tr>
                    <td>${file.id}</td>
                    <td>${file.filename}</td>
                    <td>${file.owner.name} (${file.owner.email})</td>
                    <td>${new Date(file.upload_time).toLocaleString()}</td>
                    <td>${(file.file_size / 1024).toFixed(2)} KB</td>
                    <td>${file.status}</td>
                </tr>
            `;
            filesTableBody.innerHTML += row;
        });

        // Pagination
        filesPagination.innerHTML = '';
        if (data.pages > 1) {
            for (let i = 1; i <= data.pages; i++) {
                const li = document.createElement('li');
                li.className = `page-item ${i === data.current_page ? 'active' : ''}`;
                const a = document.createElement('a');
                a.className = 'page-link';
                a.href = '#';
                a.textContent = i;
                a.addEventListener('click', (e) => {
                    e.preventDefault();
                    fetchFiles(i);
                });
                li.appendChild(a);
                filesPagination.appendChild(li);
            }
        }
    }

    fetchUsers();
    fetchFiles();
});
