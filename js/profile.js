(function() {
    "use strict";

    const Profile = (profileDiv) => {
        const name = profileDiv.dataset.name;
        const loc = profileDiv.dataset.location;

        if (name === undefined) {
            return
        }

        const img = document.createElement("img");
        img.setAttribute('src', `./images/profiles/${name.split(' ')[0].toLowerCase()}.jpg`);
        img.classList.add('profile');

        const p_name = document.createElement("p");
        p_name.classList.add("mb-0");
        p_name.innerText = name;

        profileDiv.appendChild(img);
        profileDiv.appendChild(p_name);

        if (loc !== undefined) {
            const p_loc = document.createElement("p");
            p_loc.innerText = `Currently @ ${loc}`
            profileDiv.appendChild(p_loc);
        }
    }

    document.querySelectorAll('div.contrib-profile').forEach((profileDiv) => {
        Profile(profileDiv);
    });
})();